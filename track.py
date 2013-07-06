#!/usr/bin/env python

import sys

import numpy
import SimpleCV as scv
import scipy.ndimage


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
            wim.drawCircle(c, 3, scv.Color.RED, -1)
    wim.show()


def make_background(c, n=10):
    assert n > 1
    n = float(n)
    bg = c.getImage() / n
    for _ in xrange(1, int(n)):
        bg += c.getImage() / n
    return bg


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
            yield im, fi
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
    # order them as a, b, c
    # use center of frame for ordering
    return f['corners']


def get_corners(im, test=False):
    """poll_user for corners"""
    vim = im.copy()
    corners = []
    d = scv.Display()
    d.writeFrame(vim)
    while d.isNotDone():
        p = d.leftButtonUpPosition()
        if p is not None:
            corners.append(p)
            vim.drawCircle(p, 3, scv.Color.RED, -1)
            vim.show()
            print "Corner at x:%s y%s" % p
            if len(corners) == 3:
                break
        scv.time.sleep(0.05)
    if test:
        test_box(corners, vim)
    return corners


def test_in_box(corners, p):
    a, b, c = corners
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
    u = (d11 * d02 - d01 * d12) * invd
    v = (d00 * d12 - d01 * d02) * invd
    q = -1
    if u < 0.5:
        if v < 0.5:
            q = 1
        elif v < 1.0:
            q = 4
    elif u < 1.0:
        if v < 0.5:
            q = 2
        elif v < 1.0:
            q = 3
    return q, u, v


def test_box(corners, im):
    d = scv.Display()
    d.writeFrame(im)
    while d.isNotDone():
        p = d.leftButtonUpPosition()
        if p is not None:
            q, u, v = test_in_box(corners, p)
            print q, u, v
        if d.mouseRight:
            break


def process_file(fn):
    c = scv.VirtualCamera(fn, 'video')
    bg = make_background(c)
    corners = get_corners(bg)
    f = {'bg': bg, 'corners': corners}
    for (i, (im, fi)) in enumerate(frames(c, numbers=True)):
        f['im'] = im
        f['fi'] = fi
        #f['dim'] = local_norm(bg_subtract(f['im'], f['bg']), 7, 11)
        #f['t'] = 128
        f['dim'] = bg_subtract(f['im'], f['bg'])
        #f['t'] = f['dim'].maxValue() / 2.
        f['t'] = 20
        f['bim'] = f['dim'].binarize(f['t'])
        # dilate then erode because of inversion
        f['eim'] = f['bim'].dilate(5).erode(10)
        f['blobs'] = blobs(f['eim'].invert(), 100)
        if i % 100 == 0:
            print "processing frame %04i" % i
        yield f


def show_file(fn, everyn=1):
    for (i, f) in enumerate(process_file(fn)):
        if i % everyn == 0:
            display(f)
            yield f
            #scv.time.sleep(0.05)


if __name__ == '__main__':
    fn = 'session.mov'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    en = 1
    if len(sys.argv) > 2:
        en = int(sys.argv[2])
    print "showing %s" % fn
    for _ in show_file(fn, en):
        pass
