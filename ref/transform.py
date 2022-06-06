import cv2 as cv
import numpy as np

clear = cv.imread('fingerprints/clear.png', 0)


def update(_):
    finger = cv.getTrackbarPos('finger', 'original')
    image = cv.getTrackbarPos('image', 'original')
    img = cv.imread(f'fingerprints/finger-{finger}/{image:02d}.png', 0)

    cv.imshow('original', img)

    active = cv.getTrackbarPos('active', 'calibrate')
    if active:
        img = img - clear
    cv.imshow('calibrate', img)

    h = cv.getTrackbarPos('h', 'denoise') / 10
    img = cv.fastNlMeansDenoising(img, h=h)
    cv.imshow('denoise', img)

    scale = cv.getTrackbarPos('scale', 'scale')
    img = cv.resize(img, None, fx=scale, fy=scale)
    cv.imshow('scale', img)

    gaussian_active = cv.getTrackbarPos('gaussianActive', 'threshold')
    block_size = cv.getTrackbarPos('blockSize', 'threshold') * 2 - 1
    c = cv.getTrackbarPos('C', 'threshold')
    img = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C if gaussian_active else
        cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, block_size, c)
    cv.imshow('threshold', 255 - img)

    kernel_size = cv.getTrackbarPos('kernelSize', 'morphology')
    open_active = cv.getTrackbarPos('openActive', 'morphology')
    close_active = cv.getTrackbarPos('closeActive', 'morphology')
    invert_active = cv.getTrackbarPos('closeActive', 'morphology')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if invert_active:
        if open_active:
            img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        if close_active:
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    else:
        if close_active:
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        if open_active:
            img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    cv.imshow('morphology', 255 - img)


cv.namedWindow('original', cv.WINDOW_NORMAL)
cv.namedWindow('calibrate', cv.WINDOW_NORMAL)
cv.namedWindow('denoise', cv.WINDOW_NORMAL)
cv.namedWindow('scale', cv.WINDOW_NORMAL)
cv.namedWindow('threshold', cv.WINDOW_NORMAL)
cv.namedWindow('morphology', cv.WINDOW_NORMAL)

cv.createTrackbar('finger', 'original', 1, 3, update)
cv.setTrackbarMin('finger', 'original', 1)
cv.createTrackbar('image', 'original', 0, 99, update)
cv.createTrackbar('active', 'calibrate', 1, 1, update)
cv.createTrackbar('h', 'denoise', 30, 300, update)
cv.createTrackbar('scale', 'scale', 5, 10, update)
cv.setTrackbarMin('scale', 'scale', 1)
cv.createTrackbar('gaussianActive', 'threshold', 0, 1, update)
cv.createTrackbar('blockSize', 'threshold', 30, 50, update)
cv.setTrackbarMin('blockSize', 'threshold', 2)
cv.createTrackbar('C', 'threshold', 2, 10, update)
cv.setTrackbarMin('C', 'threshold', -10)
cv.createTrackbar('kernelSize', 'morphology', 3, 10, update)
cv.createTrackbar('openActive', 'morphology', 1, 1, update)
cv.createTrackbar('closeActive', 'morphology', 1, 1, update)
cv.createTrackbar('invertActive', 'morphology', 0, 1, update)

update(None)

while True:
    if cv.waitKey() & 0xff == 27:
        break

cv.destroyAllWindows()
