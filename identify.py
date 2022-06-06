import cv2 as cv
import numpy as np

folder_path = "/home/mpi3d/Documents/sigfm-cpp/fingerprints/"
ext = ".png"

clear = cv.imread("/home/mpi3d/Documents/sigfm-cpp/fingerprints/clear.png", 0)


def update(_):
    finger_1 = cv.getTrackbarPos("finger", "image 1")
    number_1 = cv.getTrackbarPos("image", "image 1")
    image_1 = cv.imread(
        folder_path + "finger-" + str(finger_1) + "/" + str(number_1) + ext,
        cv.IMREAD_GRAYSCALE)

    if image_1 is None:
        return

    finger_2 = cv.getTrackbarPos("finger", "image 2")
    number_2 = cv.getTrackbarPos("image", "image 2")
    image_2 = cv.imread(
        folder_path + "finger-" + str(finger_2) + "/" + str(number_2) + ext,
        cv.IMREAD_GRAYSCALE)

    if image_2 is None:
        return

    if finger_1 == finger_2 and number_1 == number_2:
        return

    image_1 = 256 - clear + image_1
    image_1 = cv.normalize(image_1,
                           image_1,
                           255,
                           0,
                           norm_type=cv.NORM_MINMAX,
                           dtype=cv.CV_8U)

    image_2 = 256 - clear + image_2
    image_2 = cv.normalize(image_2,
                           image_2,
                           255,
                           0,
                           norm_type=cv.NORM_MINMAX,
                           dtype=cv.CV_8U)

    cv.imshow("image 1", image_1)
    cv.imshow("image 2", image_2)

    sift = cv.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)

    distance_match = cv.getTrackbarPos("distance match", "match") / 100

    matches_in = cv.BFMatcher().knnMatch(descriptors_1, descriptors_2, 2)

    matches_out = []
    for match_in in matches_in:
        if match_in[0].distance < distance_match * match_in[1].distance:
            match_out = (keypoints_1[match_in[0].queryIdx].pt,
                         keypoints_2[match_in[0].trainIdx].pt)
            if match_out not in matches_out:
                matches_out.append(match_out)

    print(f"Count 1: {len(matches_out)}")  # TODO Remove

    length_match = cv.getTrackbarPos("length match", "match") / 1000

    angles = []
    length = len(matches_out)
    for i in range(length):
        match_1 = matches_out[i]
        for j in range(i + 1, length):
            match_2 = matches_out[j]

            vector_1 = (match_1[0][0] - match_2[0][0],
                        match_1[0][1] - match_2[0][1])
            vector_2 = (match_1[1][0] - match_2[1][0],
                        match_1[1][1] - match_2[1][1])

            length_1 = np.sqrt(vector_1[0]**2 + vector_1[1]**2)
            length_2 = np.sqrt(vector_2[0]**2 + vector_2[1]**2)

            if length_1 > length_2: length_1, length_2 = length_2, length_1

            if 1 - length_1 / length_2 <= length_match:
                angles.append((np.arctan2(
                    vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
                    vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]),
                               match_1, match_2))

    print(f"Count 2: {len(angles)}")  # TODO Remove

    max_count = 0
    max_true_matchs = []
    angle_match = cv.getTrackbarPos("angle match", "match") * np.pi / 180

    for i, match_1 in enumerate(angles):
        count = 0
        true_matchs = list(match_1[1:])
        for j, match_2 in enumerate(angles):
            if i == j:
                continue

            distance = abs(match_1[0] - match_2[0])
            if distance <= angle_match or 2 * np.pi - distance <= angle_match:
                count += 1
                true_matchs.extend(match_2[1:])

        if count > max_count:
            max_count = count
            max_true_matchs = true_matchs

    print(max_count)  # TODO Remove
    for match in max_true_matchs:
        print("[(" + str(match[0][0]) + ", " + str(match[0][1]) + "), (" +
              str(match[1][0]) + ", " + str(match[1][1]) + ")]")  # TODO Remove

    image_3 = np.concatenate((image_1, image_2), axis=1)
    image_3 = cv.cvtColor(image_3, cv.COLOR_GRAY2RGB)

    for match_in in matches_out:
        color = (0, 255, 0) if match_in in max_true_matchs else (0, 0, 255)

        cv.line(
            image_3, (round(match_in[0][0]), round(match_in[0][1])),
            (round(match_in[1][0]) + image_1.shape[1], round(match_in[1][1])),
            color, 1, cv.LINE_AA)
        cv.circle(image_3, (round(match_in[0][0]), round(match_in[0][1])), 3,
                  color, 1, cv.LINE_AA)
        cv.circle(
            image_3,
            (round(match_in[1][0]) + image_1.shape[1], round(match_in[1][1])),
            3, color, 1, cv.LINE_AA)

    color = (0, 255,
             0) if max_count >= cv.getTrackbarPos("min match", "match") else (
                 0, 0, 255)
    cv.rectangle(image_3, (0, 0), (image_3.shape[1] - 1, image_3.shape[0] - 1),
                 color, 1, cv.LINE_AA)

    cv.imshow("match", image_3)


if clear is None:
    raise ValueError

cv.namedWindow("image 1", cv.WINDOW_NORMAL)
cv.namedWindow("image 2", cv.WINDOW_NORMAL)
cv.namedWindow("match", cv.WINDOW_NORMAL)

cv.createTrackbar("finger", "image 1", 0, 9, update)
cv.createTrackbar("image", "image 1", 0, 99, update)
cv.createTrackbar("finger", "image 2", 0, 9, update)
cv.createTrackbar("image", "image 2", 0, 99, update)
cv.createTrackbar("distance match", "match", 0, 100, update)
cv.createTrackbar("min match", "match", 0, 50, update)
cv.createTrackbar("length match", "match", 0, 1000, update)
cv.createTrackbar("angle match", "match", 0, 180, update)

cv.setTrackbarPos("image", "image 1", 0)
cv.setTrackbarPos("image", "image 2", 1)
cv.setTrackbarPos("distance match", "match", 75)
cv.setTrackbarPos("min match", "match", 5)
cv.setTrackbarPos("length match", "match", 10)
cv.setTrackbarPos("angle match", "match", 2)

update(None)

while True:
    if cv.waitKey() & 0xff == 27:
        break

cv.destroyAllWindows()
