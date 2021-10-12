from random import randint

import cv2 as cv
import numpy as np

clear = cv.imread('fingerprints/clear.png', 0)


def update(_):
    finger_1 = cv.getTrackbarPos('finger', 'image 1')
    image = cv.getTrackbarPos('image', 'image 1')
    img1 = cv.imread(f'fingerprints/finger-{finger_1}/{image:02d}.png', 0)
    finger_2 = cv.getTrackbarPos('finger', 'image 2')
    image = cv.getTrackbarPos('image', 'image 2')
    img2 = cv.imread(f'fingerprints/finger-{finger_2}/{image:02d}.png', 0)

    img1 = img1 - clear
    a = np.min(img1)
    b = np.max(img1)
    img1 = np.uint8(np.around((255 / (b - a)) * img1 - ((255 * a) / (b - a))))
    img2 = img2 - clear
    a = np.min(img2)
    b = np.max(img2)
    img2 = np.uint8(np.around((255 / (b - a)) * img2 - ((255 * a) / (b - a))))

    img1 = cv.resize(img1, None, fx=2, fy=2)
    img2 = cv.resize(img2, None, fx=2, fy=2)

    cv.imshow('image 1', img1)
    cv.imshow('image 2', img2)

    img3 = np.concatenate((img1, img2), axis=1)
    img3 = cv.cvtColor(img3, cv.COLOR_GRAY2RGB)

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    height, width = img1.shape

    d = cv.getTrackbarPos('d', 'match') / 100
    goods = {}
    for m, n in matches:
        if m.distance < d * n.distance:
            x1, y1 = kp1[m.queryIdx].pt
            x2, y2 = kp2[m.trainIdx].pt
            x2 += width
            goods[((x1, y1), (x2, y2))] = []

    min_match = cv.getTrackbarPos('minMatch', 'match')

    if len(goods) < min_match:
        print('Not enough matches')
        cv.imshow('match', img3)
        return

    pixel_match = cv.getTrackbarPos('pixelMatch', 'match') / 100
    xm = pixel_match * width
    ym = pixel_match * height

    for i1 in goods:
        x1 = i1[1][0] - i1[0][0]
        y1 = i1[1][1] - i1[0][1]

        for i2 in goods:
            if i1 == i2:
                continue

            x2 = i2[1][0] - i2[0][0]
            y2 = i2[1][1] - i2[0][1]

            if x1 - xm < x2 < x1 + xm and y1 - ym < y2 < y1 + ym:
                goods[i1].append(i2)

    goods = goods[max(goods, key=lambda i: len(goods[i]))]

    if len(goods) < min_match:
        cv.imshow('match', img3)
        return

    for good in goods:
        color = (randint(0, 255), randint(0, 255), randint(0, 255))

        cv.line(img3, (round(good[0][0]), round(good[0][1])),
                (round(good[1][0]), round(good[1][1])), color, 1, cv.LINE_AA)
        cv.circle(img3, (round(good[0][0]), round(good[0][1])), 3, color, 1,
                  cv.LINE_AA)
        cv.circle(img3, (round(good[1][0]), round(good[1][1])), 3, color, 1,
                  cv.LINE_AA)

    cv.imshow('match', img3)

    if finger_1 != finger_2:
        print(
            'Fingers must be the same. This mean that you found a false positive'
        )


cv.namedWindow('image 1', cv.WINDOW_NORMAL)
cv.namedWindow('image 2', cv.WINDOW_NORMAL)
cv.namedWindow('match', cv.WINDOW_NORMAL)

cv.createTrackbar('finger', 'image 1', 0, 9, update)
cv.createTrackbar('image', 'image 1', 0, 99, update)
cv.createTrackbar('finger', 'image 2', 0, 9, update)
cv.createTrackbar('image', 'image 2', 1, 99, update)
cv.createTrackbar('d', 'match', 75, 100, update)
cv.createTrackbar('minMatch', 'match', 5, 50, update)
cv.createTrackbar('pixelMatch', 'match', 5, 100, update)

update(None)

while True:
    if cv.waitKey() & 0xff == 27:
        break

cv.destroyAllWindows()
