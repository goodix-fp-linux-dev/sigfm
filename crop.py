import os

import cv2 as cv

WIDTH = 64
HEIGHT = 80

for root, dirs, files in os.walk("fingerprints"):
    for file in files:
        if file.endswith(".png"):
            path = os.path.join(root, file)
            img = cv.imread(path, 0)
            img = img[0:HEIGHT, 0:WIDTH]
            cv.imwrite(path, img)
