#!/usr/bin/env python

import cv2
import numpy as np
from hw1 import detectText


def binaryComponents(img):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
	dilation = cv2.dilate(img, kernel)
	erosion = cv2.erode(dilation, kernel)	
	return erosion

def main():
	img = cv2.imread('text.bmp')
	output = img.copy()
	textImg = detectText(img)
	binComponents = binaryComponents(textImg)
	h, w = binComponents.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	for i in range(0, h):
		for j in range(0, w):
			mask[:] = 0
			if binComponents[i][j] == 255:	
				_, rect = cv2.floodFill(binComponents, mask, (j, i), 0)
				cv2.rectangle(output, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))
	cv2.imshow('Words in rectangles', output)
	k = cv2.waitKey(0)
       	if k == 27:   	
		cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
