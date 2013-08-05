#!/usr/bin/env python

import logging
import os
import cPickle as pickle
import sys

import numpy
import pylab
import SimpleCV as scv
import scipy.ndimage
import math

try:
    import pygame
    nopygame = False
except ImportError:
    nopygame = True

# TODO
# - add 2nd window for binary animal image
# - work on head/tail swapping
# - add color swap option back in


# threshold for binarizing image after background subtraction
binary_threshold = 23

# dilation and erosion for removing small blobs (dirt, poo, etc...)
# when finding the animal
n_dilate = 1
n_erode = 2
# minimum blob area for finding the animal
min_blob_area = 100
max_blob_area = 10000  # still included, but no body measurements

# head detection
skeleton_radius = 4
# 1/2 the number of pixels to expand blob bounding box of suspected animal
animal_w_margin = 10
animal_h_margin = 10

# threshold to max value ratio for animal detection
animal_t_ratio = 0.6

# area around candidate head and tail to calculate average brightness
head_area = 10
tail_area = 10

# distance threshold for swapping head and tail
swap_distance_threshold = 2500.

save_frames = True


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
            pylab.scatter(cx, cy, color='y')
            # draw head, body, tail
            if b.head is not None:
                x, y = b.head
                pylab.scatter(x, y, color='r')
            if b.body is not None:
                x, y = b.body
                pylab.scatter(x, y, color='g')
            if b.tail is not None:
                x, y = b.tail
                pylab.scatter(x, y, color='b')
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
    pylab.show()
    pylab.gcf().canvas.draw()
    if save_frames:
        pylab.savefig('/Users/Catherine/Desktop/Frames/%05i.png' % f['i'])


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
            #b.show()
    if ('animal' in f) and (f['animal'] is not None):
        a = f['animal']
        wim.drawCircle(a.head, 3, scv.Color.RED, -1)
        wim.drawCircle(a.body, 3, scv.Color.GREEN, -1)
        wim.drawCircle(a.tail, 3, scv.Color.BLUE, -1)
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
    distCurv = 10
    dataLength = len(c)
    cv = numpy.empty((1, dataLength))
    cv[:] = numpy.nan #may just need to make cv an empty set
    xOutline = c[:,0]
    yOutline = c[:,1]

    for i in xrange(dataLength):
        index1 = i
        index2 = index1 + distCurv
        index3 = index1 - distCurv
        length1 = len(c)
        if index2 <= length1:
            c2 = c[index2-1,:]
        elif index2 > length1:
            indexMove = length1 - index1
            indexNo = distCurv - indexMove
            c2 = c[indexNo-1, :]

        if index3 >= 1:
            c3 = c[index3-1,:]
        elif index3 < 1:
            indexNo = length1 + index3
            c3 = c[indexNo-1, :]
        p1 = c[i-1,:]
        p2 = c2
        p3 = c3
        angle1 = math.atan2((p2[0]-p1[0]),(p2[1]-p1[1])) - math.atan2((p3[0]-p1[0]),(p3[1]-p1[1]))
        cv[0, i] = angle1
    cv = numpy.unwrap(cv) - numpy.mean(numpy.unwrap(cv))
    return cv

def find_tail_head(contour):
    c = numpy.array(contour)
    cv = measure_curvature(c)
    idxMax = numpy.argmax(cv)
    #idxMax = idxMax0[0]
    cv2 = measure_curvature(c)
    cv2[0, idxMax-1] = 0
    length1 = len(c)
    cv2[0, length1-1] = 0
    i1 = idxMax
    distCurv = len(cv[0]) / 3
    i2 = i1 + distCurv
    if i2 <= length1:
        cv2[0, i1:i2-1] = 0
    elif i2 > length1:
        iMove = length1 - i1
        iNo = distCurv - iMove
        cv2[0, i1:length1-1] = 0
        cv2[0, 0:iNo-1] = 0
    i3 = i1 - distCurv
    if i3 >= 1:
        cv2[0, i3:i1-1] = 0
    elif i3 < 1:
        iNo = length1 + i3
        cv2[0, 0:i1] = 0
        cv2[0, iNo:length1] = 0
    idxMax2 = numpy.argmax(cv2)
    #idxMax2 = idxMax2[0, 0] # What is this step for?
    tail = contour[idxMax-1]
    head = contour[idxMax2-1]
    return tail, head

def find_body(blob, tail):
    sk = blob.blobImage().skeletonize(skeleton_radius)
    # order points by distance from head: furthest to closest
    pts = numpy.array(numpy.where(sk.getGrayNumpy())).T
    tail = numpy.array(tail) - blob.topLeftCorner()
    dists = numpy.sum((pts - tail) ** 2., 1)
    spts = pts[dists.argsort()[::-1]]
    bi = len(spts) / 2
    #bi = numpy.where(spts < spts.max() / 2.)[0][-1]
    body = spts[bi]
    return body + blob.topLeftCorner()

def process_body(blob):
    tail, head = find_tail_head(blob.contour())
    body = find_body(blob, tail)
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
        #scv.time.sleep(0.05)
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
    d['e'] = b.e
    d['z'] = b.z
    return d


def cull_possible_animals(ps, la):
    if la is None:
        # return biggest?
        return max(ps, key=lambda i: i.area())
    # return closest TODO should this use the centroid?
    return min(ps, key=lambda i: (la.x - i.x) ** 2. + (la.y - i.y) ** 2.)


def refine_animal(a, f, l, swap_distance=True, swap_color=True):
    """
    a : animal 'blob'
    f : frame features
    """
    if a is None:
        return None
    # crop image to a.bb + some margin
    cim = f['dim'].crop(a.x, a.y, a.width() + animal_w_margin * 2,
                        a.height() + animal_h_margin * 2, centered=True)

    # re-threshold
    abim = cim.binarize(cim.maxValue() * animal_t_ratio).invert()
    # get biggest blob
    blobs = abim.findBlobs()
    if blobs is None:
        print "Failed to refind animal"
        return a
    if len(blobs) > 1:
        blob = max(blobs, key=lambda b: b.area())
    elif len(blobs) == 1:
        blob = blobs[0]
    else:
        print "Failed to refind animal"
        return a

    # find head & tail
    h, b, t = process_body(blob)

    # offset blob h, b, t
    o = a.topLeftCorner() - numpy.array([animal_w_margin, animal_h_margin])
    a.head = h + o
    a.body = b + o
    a.tail = t + o

    # fix head and tail (head should be darker)
    him = f['im'].crop(a.head[0], a.head[1], head_area, head_area, centered=True)
    tim = f['im'].crop(a.tail[0], a.tail[1], tail_area, tail_area, centered=True)
    if (l is not None) and hasattr(l, 'head') and hasattr(l, 'tail'):
        dh = ((l.head[0] - a.head[0]) ** 2. + (l.head[1] - a.head[1]) ** 2.)
        dt = ((l.head[0] - a.tail[0]) ** 2. + (l.head[1] - a.tail[1]) ** 2.)
    else:
        dh = -1
        dt = -1
    #print dh, dt

    if (swap_distance and (abs(dh - dt) > swap_distance_threshold) and (dt < dh)):
        a.head, a.tail = a.tail, a.head
        #print 'swap'
    else:
        if (swap_color and (abs(dh - dt) < swap_distance_threshold) and (dt < dh) and (him.meanColor()[0] > tim.meanColor()[0])):
            a.head, a.tail = a.tail, a.head
            #print 'swap2'
    return a


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
    last_animal = None
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
            possible_animals = []
            if f['blobs'] is not None:
                for b in f['blobs']:
                    q, u, v = test_in_box(f['corners'], b.centroid())
                    b.quadrant = q
                    b.u = u
                    b.v = v
                    if q != -1:
                        possible_animals.append(b)
                    if q != -1 and b.area() < max_blob_area:
                        head, body, tail = process_body(b)
                        b.head = head
                        b.body = body
                        b.tail = tail
                    else:
                        b.head = None
                        b.body = None
                        b.tail = None
                    if b.head is None:
                        b.e = None
                        b.z = None
                    else:
                        _, e, z = test_in_box(f['corners'], (b.head[0], b.head[1]))
                        b.e = e
                        b.z = z
            if len(possible_animals) == 1:
                animal = possible_animals[0]
            elif len(possible_animals) == 0:
                animal = None
            else:
                animal = cull_possible_animals(possible_animals, last_animal)
            animal = refine_animal(animal, f, last_animal)
            f['animal'] = animal
            if i % 100 == 0:
                print "processing frame %04i" % i
            d = dict([(k, f[k]) for k in ['fi', 'i', 't']])
            if f['blobs'] is None:
                d['blobs'] = []
            else:
                d['blobs'] = [blob_to_dict(b) for b in f['blobs']]
            if f['animal'] is None:
                d['animal'] = None
            else:
                d['animal'] = blob_to_dict(f['animal'])
            rs.append(d)
            #rs.append(dict([(k, f[k]) for k in ['fi', 'i', 't', 'blobs']]))
            if i % save_every_n == 0:
                with open(rfn, 'w') as rf:
                    pickle.dump(rs, rf)
            yield f
            if animal is not None:
                last_animal = animal
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
            scv.time.sleep(0.8)


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
