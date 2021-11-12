import os

import cv2 as cv
import numpy as np


def compute_finger(images_path: str, clear_path: str, save_path: str) -> None:
    finger = []
    sift = cv.SIFT_create()
    clear: np.ndarray = cv.imread(clear_path, 0)
    for root, _, files in os.walk(images_path):
        for file in files:
            if file.endswith(".png"):
                image: np.ndarray = cv.imread(os.path.join(root, file),
                                              0) - clear

                minimum: int = np.min(image)
                maximum: int = np.max(image)
                tmp = 255 / (maximum - minimum)
                image = np.uint8(np.round(tmp * image - minimum * tmp))

                keypoints: tuple
                descriptors: np.ndarray
                keypoints, descriptors = sift.detectAndCompute(image, None)

                length = len(keypoints)
                data = np.empty((2, length), np.ndarray)
                for i in range(length):
                    data[0, i] = keypoints[i].pt
                    data[1, i] = descriptors[i]

                finger.append(data)

    np.savez_compressed(save_path, *finger)


if __name__ == "__main__":
    compute_finger('fingerprints/finger-0', 'fingerprints/clear.png',
                   'fingerprints/finger-0.npz')
