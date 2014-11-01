#!/usr/bin/env python

import cv2
import numpy as np

def detectText(img):
	smoothImg = cv2.GaussianBlur(img, (3,3), 0.1)
	gray = cv2.cvtColor(smoothImg, cv2.COLOR_BGR2GRAY)
	laplac = cv2.Laplacian(gray, ddepth = cv2.CV_16S, ksize = 3, scale = 1, delta = 0)
	laplac = cv2.convertScaleAbs(laplac)
	(thresh, dst) = cv2.threshold(laplac, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	return dst

def main():
	img = cv2.imread('text.bmp')
	cv2.imshow('Text detection', detectText(img))
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

