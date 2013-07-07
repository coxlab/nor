#!/usr/bin/env python

import logging
import os
import cPickle as pickle
import sys

import numpy
import SimpleCV as scv
import scipy.ndimage


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


def frames(c, numbers=False):
    if numbers:
        fi = c.getFrameNumber()
        im = c.getImage()
        while im is not None:
            yield fi, im
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
    return d


def process_file(fn):
    c = scv.VirtualCamera(fn, 'video')
    bg = load_background(fn)
    if bg is None:
        bg = make_background(c)
        save_background(bg, fn)
        print "You have to restart this file to get the correct frame numbers"
        sys.exit(1)
    bg = make_background(c)
    corners = load_corners(fn)
    if corners is None:
        corners = get_corners(bg)
        save_corners(corners, fn)
    f = {'bg': bg, 'corners': corners}
    f['corners'] = order_corners(f)
    rfn = os.path.splitext(fn)[0] + '_track.p'
    rs = []
    try:
        for (i, (fi, im)) in enumerate(frames(c, numbers=True)):
            f['im'] = im
            f['fi'] = fi
            f['i'] = i
            #f['dim'] = local_norm(bg_subtract(f['im'], f['bg']), 7, 11)
            #f['t'] = 128
            f['dim'] = bg_subtract(f['im'], f['bg'])
            #f['t'] = f['dim'].maxValue() / 2.
            f['t'] = 20
            f['bim'] = f['dim'].binarize(f['t'])
            # dilate then erode because of inversion
            f['eim'] = f['bim'].dilate(5).erode(10)
            f['blobs'] = blobs(f['eim'].invert(), 100)
            if f['blobs'] is not None:
                for b in f['blobs']:
                    q, u, v = test_in_box(f['corners'], b.centroid())
                    b.quadrant = q
                    b.u = u
                    b.v = v
            if i % 100 == 0:
                print "processing frame %04i" % i
            d = dict([(k, f[k]) for k in ['fi', 'i', 't']])
            if f['blobs'] is None:
                d['blobs'] = []
            else:
                d['blobs'] = [blob_to_dict(b) for b in f['blobs']]
            rs.append(d)
            rs.append(dict([(k, f[k]) for k in ['fi', 'i', 't', 'blobs']]))
            yield f
    except KeyboardInterrupt as E:
        # save results
        logging.debug("Saving %i results to %s" % (len(rs), rfn))
        with open(rfn, 'w') as f:
            pickle.dump(rs, f)
        # re-raise
        raise E
    logging.debug("Saving %i results to %s" % (len(rs), rfn))
    with open(rfn, 'w') as f:
        pickle.dump(rs, f)


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
