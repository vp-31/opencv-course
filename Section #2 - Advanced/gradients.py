#pylint:disable=no-member

import cv2 as cv
import numpy as np
import cv_utilities as vcv

import os
import sys
os.chdir(os.path.split(sys.argv[0])[0])


img = cv.imread('../Resources/Photos/park.jpg')
# cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

vcv.imshow_hstack_windowSize('img, gray', [img, gray])

# Laplacian
lap = cv.Laplacian(gray, -1)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

lap_CV_64F = cv.Laplacian(gray, cv.CV_64F)
lap_CV_64F = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

canny = cv.Canny(gray, 150, 175)
# cv.imshow('Canny', canny)
vcv.imshow_hstack_windowSize('lap, lap_CV_64F, canny', [lap, lap_CV_64F, canny])

# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

# cv.imshow('Sobel X', sobelx)
# cv.imshow('Sobel Y', sobely)
# cv.imshow('Combined Sobel', combined_sobel)

vcv.imshow_hstack_windowSize('sobelx, sobely, combined_sobel', [sobelx, sobely, combined_sobel])
cv.waitKey(0)