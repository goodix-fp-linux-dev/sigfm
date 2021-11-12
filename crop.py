import os

import cv2 as cv
import numpy as np

WIDTH = 64
HEIGHT = 80

for root, _, files in os.walk("fingerprints"):
    for file in files:
        if file.endswith(".png"):
            path = os.path.join(root, file)
            image: np.ndarray = cv.imread(path, 0)
            if image.shape != (HEIGHT, WIDTH):
                cv.imwrite(path, image[0:HEIGHT, 0:WIDTH])
