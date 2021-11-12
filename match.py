from time import time

import cv2 as cv
import numpy as np

DISTANCE_MATCH = 0.75
LENGTH_MATCH = 0.05
ANGLE_MATCH = 0.05
MIN_MATCH = 5


def match(image_path: str, clear_path: str, sample_path) -> bool:
    image: np.ndarray = cv.imread(image_path, 0) - cv.imread(clear_path, 0)

    minimum: int = np.min(image)
    maximum: int = np.max(image)
    tmp = 255 / (maximum - minimum)
    image = np.uint8(np.round(tmp * image - minimum * tmp))

    keypoints_1: tuple
    descriptors_1: np.ndarray
    keypoints_1, descriptors_1 = cv.SIFT_create().detectAndCompute(image, None)

    if len(keypoints_1) < MIN_MATCH:
        return False

    matcher = cv.BFMatcher()
    with np.load(sample_path, allow_pickle=True) as samples:
        for name in samples:
            keypoints_2: np.ndarray
            descriptors_2: np.ndarray
            keypoints_2, descriptors_2 = samples[name]

            if len(keypoints_2) < MIN_MATCH:
                continue

            descriptors_2 = np.stack(descriptors_2)
            points: tuple = matcher.knnMatch(descriptors_1, descriptors_2, k=2)

            if len(points) < MIN_MATCH:
                continue

            matchs = []
            for match_1, match_2 in points:
                if match_1.distance < DISTANCE_MATCH * match_2.distance:
                    matchs.append((keypoints_1[match_1.queryIdx].pt,
                                   keypoints_2[match_1.trainIdx]))

            matchs = list(set(matchs))

            if len(matchs) < MIN_MATCH:
                continue

            angles = []
            for i, match_1 in enumerate(matchs):
                for match_2 in matchs[i + 1:]:
                    vec_1 = (match_1[0][0] - match_2[0][0],
                             match_1[0][1] - match_2[0][1])
                    vec_2 = (match_1[1][0] - match_2[1][0],
                             match_1[1][1] - match_2[1][1])

                    length_1 = np.sqrt(vec_1[0]**2 + vec_1[1]**2)
                    length_2 = np.sqrt(vec_2[0]**2 + vec_2[1]**2)

                    if 1 - min(length_1, length_2) / max(
                            length_1, length_2) <= LENGTH_MATCH:
                        product = length_1 * length_2
                        angles.append(
                            (np.pi / 2 + np.arcsin(
                                (vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) /
                                product),
                             np.arccos(
                                 (vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]) /
                                 product)))

            if len(angles) < MIN_MATCH:
                continue

            for i, angle_1 in enumerate(angles):
                count = 0
                for j, angle_2 in enumerate(angles):
                    if i == j:
                        continue

                    if 1 - min(angle_1[0], angle_2[0]) / max(
                            angle_1[0], angle_2[0]) <= ANGLE_MATCH and 1 - min(
                                angle_1[1], angle_2[1]) / max(
                                    angle_1[1], angle_2[1]) <= ANGLE_MATCH:
                        count += 1

                    if count >= MIN_MATCH:
                        return True

        return False


if __name__ == '__main__':
    start = time()
    print(
        match('fingerprints/fingerprint.png', 'fingerprints/clear.png',
              'fingerprints/finger-2.npz'))
    print(f"Time: {time() - start}")
