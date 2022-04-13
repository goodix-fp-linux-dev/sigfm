from random import randint
from sys import exit as stop

import cv2 as cv
import numpy as np

clear: np.ndarray = cv.imread("/home/mango/fpr/clear.jpg", 0)


def update(_):
    finger_1: int = cv.getTrackbarPos('finger', 'image 1')
    number_1: int = cv.getTrackbarPos('image', 'image 1')
    image_1: np.ndarray = cv.imread(
        f"/home/mango/fpr/finger-{finger_1}/"
        f"{number_1:d}.jpg", 0)

    if image_1 is None:
        return

    finger_2: int = cv.getTrackbarPos('finger', 'image 2')
    number_2: int = cv.getTrackbarPos('image', 'image 2')
    image_2: np.ndarray = cv.imread(
        f"/home/mango/fpr/finger-{finger_2}/"
        f"{number_2:d}.jpg", 0)

    if image_2 is None:
        return

    if finger_1 == finger_2 and number_1 == number_2:
        return

    minimum: int
    maximum: int

    image_1 -= clear
    minimum = np.min(image_1)
    maximum = np.max(image_1)
    tmp = 255 / (maximum - minimum)
    image_1 = np.uint8(np.around(tmp * image_1 - minimum * tmp))

    image_2 -= clear
    minimum = np.min(image_2)
    maximum = np.max(image_2)
    tmp = 255 / (maximum - minimum)
    image_2 = np.uint8(np.around(tmp * image_2 - minimum * tmp))

    cv.imshow('image 1', image_1)
    cv.imshow('image 2', image_2)

    sift = cv.SIFT_create()

    keypoints_1: tuple
    descriptors_1: np.ndarray
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)

    keypoints_2: tuple
    descriptors_2: np.ndarray
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)

    matchs = []
    distance_match: float = cv.getTrackbarPos('distance match', 'match') / 100
    for match_1, match_2 in cv.BFMatcher().knnMatch(descriptors_1,
                                                    descriptors_2,
                                                    k=2):
        if match_1.distance < distance_match * match_2.distance:
            matchs.append((keypoints_1[match_1.queryIdx].pt,
                           keypoints_2[match_1.trainIdx].pt))

    matchs = list(set(matchs))
    print(matchs)
    angles = []
    
    length_match: float = cv.getTrackbarPos('length match', 'match') / 1000
    for i, match_1 in enumerate(matchs):
        for match_2 in matchs[i + 1:]:
            vec_1 = (match_1[0][0] - match_2[0][0],
                     match_1[0][1] - match_2[0][1])
            vec_2 = (match_1[1][0] - match_2[1][0],
                     match_1[1][1] - match_2[1][1])

            length_1 = np.sqrt(vec_1[0]**2 + vec_1[1]**2)
            length_2 = np.sqrt(vec_2[0]**2 + vec_2[1]**2)

            if 1 - min(length_1, length_2) / max(length_1,
                                                 length_2) <= length_match:
                product = length_1 * length_2
                angles.append(
                    ((np.pi / 2 + 
                        np.arcsin((vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) / product),
                        np.arccos((vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]) / product)
                     ),
                     (match_1, match_2)))

    count = 0
    max_count = 0
    max_true_matchs = []
    angle_match: float = cv.getTrackbarPos('angle match', 'match') / 1000
    for i, (angle_1, match_1) in enumerate(angles):
        count = 0
        true_matchs = []
        for j, (angle_2, match_2) in enumerate(angles):
            if i == j:
                continue

            if 1 - min(angle_1[0], angle_2[0]) / max(
                    angle_1[0], angle_2[0]) <= angle_match and 1 - min(
                        angle_1[1], angle_2[1]) / max(
                            angle_1[1], angle_2[1]) <= angle_match:
                count += 1
                for match in match_1 + match_2:
                    if match not in true_matchs:
                        true_matchs.append(match)

        if count >= max_count:
            max_count = count
            max_true_matchs = true_matchs

    image_3: np.ndarray
    image_3 = np.concatenate((image_1, image_2), axis=1)
    image_3 = cv.cvtColor(image_3, cv.COLOR_GRAY2RGB)

    for match in matchs:
        color = (0, 255, 0) if match in max_true_matchs else (0, 0, 255)

        cv.line(image_3, (round(match[0][0]), round(match[0][1])),
                (round(match[1][0]) + image_1.shape[1], round(match[1][1])),
                color, 1, cv.LINE_AA)
        cv.circle(image_3, (round(match[0][0]), round(match[0][1])), 3, color,
                  1, cv.LINE_AA)
        cv.circle(image_3,
                  (round(match[1][0]) + image_1.shape[1], round(match[1][1])),
                  3, color, 1, cv.LINE_AA)

    min_match = cv.getTrackbarPos('min match', 'match')
    if max_count >= min_match:
        if finger_1 != finger_2:
            cv.destroyAllWindows()
            print(
                f"You found a false positive with image {finger_1}:{number_1} "
                f"and image {finger_2}:{number_2}")
            stop()

        cv.rectangle(image_3, (0, 0),
                     (image_3.shape[1] - 1, image_3.shape[0] - 1), (0, 255, 0),
                     1, cv.LINE_AA)
    else:
        cv.rectangle(image_3, (0, 0),
                     (image_3.shape[1] - 1, image_3.shape[0] - 1), (0, 0, 255),
                     1, cv.LINE_AA)

    cv.imshow('match', image_3)


cv.namedWindow('image 1', cv.WINDOW_NORMAL)
cv.namedWindow('image 2', cv.WINDOW_NORMAL)
cv.namedWindow('match', cv.WINDOW_NORMAL)

cv.createTrackbar('finger', 'image 1', 0, 9, update)
cv.createTrackbar('image', 'image 1', 0, 99, update)
cv.createTrackbar('finger', 'image 2', 0, 9, update)
cv.createTrackbar('image', 'image 2', 1, 99, update)
cv.createTrackbar('distance match', 'match', 75, 100, update)
cv.createTrackbar('min match', 'match', 5, 50, update)
cv.createTrackbar('length match', 'match', 50, 1000, update)
cv.createTrackbar('angle match', 'match', 50, 1000, update)

update(None)

while True:
    if cv.waitKey() & 0xff == 27:
        break

cv.destroyAllWindows()
stop()
