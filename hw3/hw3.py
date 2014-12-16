#!/usr/bin/env python

import cv2
import numpy as np

def doHighPassFiltering(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def main():
    img = cv2.imread('mandril.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imwrite("filter.bmp", doHighPassFiltering(img))
    cv2.imwrite("laplacian.bmp", cv2.Laplacian(img, cv2.CV_32F))

if __name__ == "__main__":
    main()

