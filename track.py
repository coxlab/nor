#!/usr/bin/env python

import logging
import os
import cPickle as pickle
import sys

import numpy
import pylab
import SimpleCV as scv
import scipy.ndimage

try:
    import pygame
    nopygame = False
except ImportError:
    nopygame = True


# threshold for binarizing image after background subtraction
binary_threshold = 20

# dilation and erosion for removing small blobs (dirt, poo, etc...)
# when finding the animal
n_dilate = 5
n_erode = 10
# minimum blob area for finding the animal
min_blob_area = 100
max_blob_area = 10000  # still included, but no body measurements

# head detection
skeleton_radius = 7

save_frames = False


def load_corners(fn):
    cfn = os.path.splitext(fn)[0] + '_corners.p'
    if not os.path.exists(cfn):
        return None
    logging.debug("Loading corners from: %s" % cfn)
    with open(cfn, 'r') as f:
        r = pickle.load(f)
    return r


def save_corners(corners, fn):
    cfn = os.path.splitext(fn)[0] + '_corners.p'
    logging.debug("Saving corners to: %s" % cfn)
    with open(cfn, 'w') as f:
        pickle.dump(corners, f)


def display(f):
    if nopygame is False:
        return display_cv(f)
    wim = f['im'].getGrayNumpy()
    pylab.ion()
    pylab.clf()
    pylab.imshow(wim.T, vmin=0, vmax=255, cmap=pylab.cm.gray)
    pylab.title('thresh: %i' % f['t'])
    if (f['blobs'] is not None) and len(f['blobs']):
        for (i, b) in enumerate(f['blobs']):
            # draw bounding box
            x, y, w, h = b.boundingBox()
            pylab.plot(
                [x, x + w, x + w, x, x],
                [y, y, y + h, y + h, y], color='b')
            # draw centroid
            cx, cy = b.centroid()
            pylab.scatter(cx, cy, color='b')
            # draw head, body, tail
            if b.head is not None:
                x, y = b.head
                pylab.scatter(x, y, color='r')
            if b.body is not None:
                x, y = b.body
                pylab.scatter(x, y, color='g')
            if b.tail is not None:
                x, y = b.tail
                pylab.scatter(x, y, color='m')
            if all([i is not None for i in (b.head, b.body, b.tail)]):
                pylab.plot(
                    [b.tail[0], b.body[0], b.head[0]],
                    [b.tail[1], b.body[1], b.head[1]], color='y')
            c = numpy.array(b.contour())
            pylab.plot(c[:, 0], c[:, 1], color='b')
    if 'corners' in f:
        # draw corners
        xs = []
        ys = []
        for c in f['corners']:
            xs.append(c['x'])
            ys.append(c['y'])
        pylab.scatter(xs, ys, color='r')
    #pylab.show()
    pylab.gcf().canvas.draw()
    if save_frames:
        pylab.savefig('%05i.png' % f['i'])


def display_cv(f):
    wim = f['im']
    wim.drawText('thresh: %i' % f['t'], 10, 10, scv.Color.RED)
    y = 20
    if (f['blobs'] is not None) and len(f['blobs']):
        cm = scv.ColorMap(
            (scv.Color.RED, scv.Color.YELLOW, scv.Color.BLUE),
            0, len(f['blobs']))
        for (i, b) in enumerate(f['blobs']):
            if 'corners' in f:
                q, u, v = test_in_box(f['corners'], b.centroid())
            args = b.boundingBox()
            args.append(cm[i])
            wim.drawRectangle(*args)
            wim.drawText('blob %i area %i q=%i' %
                         (i, b.area(), q), 10, y, cm[i])
            y += 10
    if 'corners' in f:
        for c in f['corners']:
            wim.drawCircle((c['x'], c['y']), 3, scv.Color.RED, -1)
            wim.drawText(str(c['i']), c['x'] + 2, c['y'], scv.Color.RED)
    wim.show()


def make_background(c, n=10):
    assert n > 1
    n = float(n)
    bg = c.getImage() / n
    for _ in xrange(1, int(n)):
        bg += c.getImage() / n
    return bg


def load_background(fn):
    bfn = os.path.splitext(fn)[0] + '_bg.png'
    if not os.path.exists(bfn):
        return None
    logging.debug("Loading background from: %s" % bfn)
    return scv.Image(bfn)


def save_background(bg, fn):
    bfn = os.path.splitext(fn)[0] + '_bg.png'
    logging.debug("Saving background to: %s" % bfn)
    bg.save(bfn)


def gaussian_blur(im, sigma):
    d = sigma * 3
    if d % 2 == 0:
        d += 1
    return im.gaussianBlur((d, d), sigma, sigma, True)


def local_norm(im, msigma, vsigma, asarray=False):
    a = im.getGrayNumpy().astype('f8')
    mim = a - scipy.ndimage.filters.gaussian_filter(a, msigma)
    vim = scipy.ndimage.filters.gaussian_filter(mim ** 2., vsigma) ** 0.5
    a = (mim / vim)
    # stretch to 0 - 255
    a = ((a - a.min()) / (a.max() - a.min()) * 255.).astype('uint8')
    if asarray:
        return a
    return scv.Image(a)


def bg_subtract(im, bg):
    return ((im - bg) + (bg - im)) / 2.


def blobs(im, minarea=None):
    b = im.findBlobs(minsize=minarea)
    if b is None:
        return b
    if minarea is None:
        return b
    #b = b.filter(b.area() > minarea)
    b = b.sortArea()[::-1]
    return b


def measure_curvature(contour):
    """
    < 0 = convex
    > 0 = concave
    """
    c = numpy.array(contour)
    i1 = [-1, ] + range(c.shape[0] - 1)
    i2 = range(c.shape[0])
    i3 = range(1, c.shape[0]) + [0, ]
    x1 = c[i1, 0]
    x2 = c[i2, 0]
    x3 = c[i3, 0]
    y1 = c[i1, 1]
    y2 = c[i2, 1]
    y3 = c[i3, 1]
    sqr = lambda i: i ** 2.
    n = 2. * ((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2))
    d = numpy.sqrt(
        (sqr(x2 - x1) + sqr(y2 - y1)) *
        (sqr(x3 - x2) + sqr(y3 - y2)) *
        (sqr(x1 - x3) + sqr(y1 - y3)))
    return n / d


def find_tail(contour):
    c = numpy.array(contour)
    rcv = measure_curvature(c)
    cc = c[rcv < 0]
    cv = measure_curvature(cc)
    # find minimum
    return cc[cv.argmin()]


def find_head(blob, tail):
    sk = blob.blobImage().skeletonize(skeleton_radius)
    # order points by distance from head: furthest to closest
    pts = numpy.array(numpy.where(sk.getGrayNumpy())).T
    tail = numpy.array(tail) - blob.topLeftCorner()
    dists = numpy.sum((pts - tail) ** 2., 1)
    spts = pts[dists.argsort()[::-1]]
    head = spts[0]
    bi = numpy.where(spts < spts.max() / 2.)[0][0]
    body = spts[bi]
    return head + blob.topLeftCorner(), body + blob.topLeftCorner()


def process_body(blob):
    tail = find_tail(blob.contour())
    head, body = find_head(blob, tail)
    return head, body, tail


def frames(c, numbers=False):
    if numbers:
        fi = c.getFrameNumber()
        im = c.getImage()
        notfirst = False
        while im is not None:
            yield fi, im
            if notfirst and c.getFrameNumber() < fi:
                break
            notfirst = True
            fi = c.getFrameNumber()
            im = c.getImage()
    else:
        im = c.getImage()
        while im is not None:
            yield im
            im = c.getImage()


def order_corners(f):
    if 'corners' not in f:
        raise AttributeError('corners missing from features')
    if 'bg' not in f:
        raise AttributeError('bg missing from features')
    # order them as a, b, c
    # use center of frame for ordering
    hw = f['bg'].width / 2.
    hh = f['bg'].width / 2.
    for c in f['corners']:
        if c['x'] < hw:  # left
            if c['y'] < hh:
                c['i'] = 0
            else:
                c['i'] = 2
        else:  # right
            if c['y'] < hh:
                c['i'] = 1
            else:
                c['i'] = 3
    inds = sorted([c['i'] for c in f['corners']])
    cbi = {}
    for c in f['corners']:
        cbi[c['i']] = c
    if inds == [0, 1, 2]:
        return [cbi[0], cbi[1], cbi[2]]
    elif inds == [0, 1, 3]:
        return [cbi[1], cbi[3], cbi[0]]
    elif inds == [1, 2, 3]:
        return [cbi[3], cbi[2], cbi[1]]
    elif inds == [0, 2, 3]:
        return [cbi[2], cbi[0], cbi[3]]
    else:
        raise Exception("Failed to sort corners: %s" % (f['corners']))


def get_corners(im, test=False):
    """poll_user for corners"""
    vim = im.copy()
    corners = []
    d = scv.Display()
    d.writeFrame(vim)
    print "Please click on three corners of the box"
    while d.isNotDone():
        p = d.leftButtonUpPosition()
        if p is not None:
            c = dict(x=p[0], y=p[1], i=-1)
            corners.append(c)
            vim.drawCircle(p, 3, scv.Color.RED, -1)
            vim.show()
            print "Corner at x:%s y%s" % p
            if len(corners) == 3:
                break
        scv.time.sleep(0.05)
    if test:
        corners = order_corners({'bg': im, 'corners': corners})
        test_box(corners, vim)
    return corners


def test_in_box(corners, p):
    a, b, c = [(c['x'], c['y']) for c in corners]
    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array(c)
    p = numpy.array(p)
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = numpy.dot(v0, v0)
    d01 = numpy.dot(v0, v1)
    d02 = numpy.dot(v0, v2)
    d11 = numpy.dot(v1, v1)
    d12 = numpy.dot(v1, v2)

    invd = 1. / (d00 * d11 - d01 * d01)
    v = (d11 * d02 - d01 * d12) * invd
    u = (d00 * d12 - d01 * d02) * invd
    q = -1
    i = corners[0]['i']
    if i == 0:
        pass
    elif i == 1:
        u, v = v, u
        u = 1. - u
    elif i == 2:
        u, v = v, u
        v = 1. - v
    elif i == 3:
        u = 1. - u
        v = 1. - v
    else:
        raise Exception("Invalid corner index: %s" % i)
    if (u > 0.0) and (v > 0.0):
        if u < 0.5:
            if v < 0.5:
                q = 0
            elif v < 1.0:
                q = 2
        elif u < 1.0:
            if v < 0.5:
                q = 1
            elif v < 1.0:
                q = 3
    return q, u, v


def test_box(corners, im):
    d = scv.Display()
    wim = im.copy()
    for c in corners:
        wim.drawCircle((c['x'], c['y']), 3, scv.Color.RED, -1)
        wim.drawText(str(c['i']), c['x'] + 2, c['y'], scv.Color.RED)
    d.writeFrame(wim)
    while d.isNotDone():
        p = d.leftButtonUpPosition()
        if p is not None:
            q, u, v = test_in_box(corners, p)
        if d.mouseRight:
            break


def blob_to_dict(b):
    d = {}
    d['x'] = b.x
    d['y'] = b.y
    d['u'] = b.u
    d['v'] = b.v
    d['quadrant'] = b.quadrant
    d['width'] = b.width()
    d['height'] = b.height()
    d['area'] = b.area()
    d['cx'], d['cy'] = b.centroid()
    d['head'] = b.head
    d['body'] = b.body
    d['tail'] = b.tail
    return d


def process_file(fn, save_every_n=1000, burn=0):
    c = scv.VirtualCamera(fn, 'video')
    if burn:
        print "skipping %i frames" % burn
        for _ in xrange(burn):
            c.getImage()
    bg = load_background(fn)
    if bg is None:
        bg = make_background(c)
        save_background(bg, fn)
        print "You have to restart this file to get the correct frame numbers"
        sys.exit(1)
    corners = load_corners(fn)
    if corners is None:
        corners = get_corners(bg)
        save_corners(corners, fn)
    f = {'bg': bg, 'corners': corners}
    f['corners'] = order_corners(f)
    rfn = os.path.splitext(fn)[0] + '_track.p'
    rs = []
    try:
        for (i, (fi, im)) in enumerate(frames(c, numbers=True), burn):
            f['im'] = im
            f['fi'] = fi
            f['i'] = i
            #f['dim'] = local_norm(bg_subtract(f['im'], f['bg']), 7, 11)
            #f['t'] = 128
            f['dim'] = bg_subtract(f['im'], f['bg'])
            #f['t'] = f['dim'].maxValue() / 2.
            f['t'] = binary_threshold
            f['bim'] = f['dim'].binarize(f['t'])
            # dilate then erode because of inversion
            f['eim'] = f['bim'].dilate(n_dilate).erode(n_erode)
            f['blobs'] = blobs(f['eim'].invert(), min_blob_area)
            if f['blobs'] is not None:
                for b in f['blobs']:
                    q, u, v = test_in_box(f['corners'], b.centroid())
                    b.quadrant = q
                    b.u = u
                    b.v = v
                    if q != -1 and b.area() < max_blob_area:
                        head, body, tail = process_body(b)
                        b.head = head
                        b.body = body
                        b.tail = tail
                    else:
                        b.head = None
                        b.body = None
                        b.tail = None
            if i % 100 == 0:
                print "processing frame %04i" % i
            d = dict([(k, f[k]) for k in ['fi', 'i', 't']])
            if f['blobs'] is None:
                d['blobs'] = []
            else:
                d['blobs'] = [blob_to_dict(b) for b in f['blobs']]
            rs.append(d)
            #rs.append(dict([(k, f[k]) for k in ['fi', 'i', 't', 'blobs']]))
            if i % save_every_n == 0:
                with open(rfn, 'w') as rf:
                    pickle.dump(rs, rf)
            yield f
    except KeyboardInterrupt as E:
        # save results
        logging.debug("Saving %i results to %s" % (len(rs), rfn))
        with open(rfn, 'w') as rf:
            pickle.dump(rs, rf)
        # re-raise
        raise E
    logging.debug("Saving %i results to %s" % (len(rs), rfn))
    with open(rfn, 'w') as rf:
        pickle.dump(rs, rf)


def show_file(fn, everyn=1):
    for (i, f) in enumerate(process_file(fn)):
        if i % everyn == 0:
            display(f)
            yield f
            #scv.time.sleep(0.05)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fn = 'session.mov'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    en = 1
    if len(sys.argv) > 2:
        en = int(sys.argv[2])
    print "showing %s" % fn
    if en < 1:
        for _ in process_file(fn):
            pass
    else:
        for _ in show_file(fn, en):
            pass
