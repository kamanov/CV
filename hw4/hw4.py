#!/usr/bin/env python

import cv2
import numpy as np

def main():
    img = cv2.imread("mandril.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    sift = cv2.SIFT()
    keypoints_img, desc_img = sift.detectAndCompute(img, None)

    #rotate 45, scale 1/2
    rows, cols = img.shape
    crow, ccol = rows/2, cols/2
    rotation_matrix = cv2.getRotationMatrix2D((crow, ccol), 45, 0.5)

    transformed_img = cv2.warpAffine(img, rotation_matrix, img.shape)
    keypoints_trans_img, desc_trans_img = sift.detectAndCompute(transformed_img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_img, desc_trans_img, k=2)

    init_points = np.array(map(lambda x: np.array(x.pt, np.float32), keypoints_img))
    transformed_points = np.array(map(lambda x: np.dot(rotation_matrix, (x[0], x[1], 1)), init_points))
    matches_cnt = 0
    for i, (m, n) in enumerate(matches):
        distance = np.linalg.norm(transformed_points[i] - keypoints_trans_img[m.trainIdx].pt)
        matches_cnt +=  1 if (distance < 50) else 0

    print "Match : %s%% " % (matches_cnt * 100.0 / len(matches))


if __name__ == "__main__":
    main()