#!/usr/bin/env python
import numpy as np
import cv2

TRACK_POINTS_NUM = 100

def harris(frame):
    return cv2.goodFeaturesToTrack(frame,
                                   useHarrisDetector=True,
                                   maxCorners=TRACK_POINTS_NUM,
                                   qualityLevel=0.15,
                                   minDistance=7,
                                   blockSize=7)

def fast(frame):
    fast = cv2.FastFeatureDetector().detect(frame)
    fast = sorted(fast, key=lambda k: -k.response)
    fast = [[[kp.pt[0], kp.pt[1]]] for kp in fast[:TRACK_POINTS_NUM]]
    fast = np.array(fast, np.float32)
    return fast
    
    
def open_videos(in_filename, out_filename):
    source_video = cv2.VideoCapture(in_filename)
    width = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(source_video.get(cv2.cv.CV_CAP_PROP_FPS))
    fourcc = int(source_video.get(cv2.cv.CV_CAP_PROP_FOURCC))
    return source_video, cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

def start(detector, out_filename):
    source , writer = open_videos('sequence.mpg', '%s.avi' % out_filename)
    
    retval, grabbed = source.read()
    grabbed = cv2.cvtColor(grabbed, cv2.COLOR_BGR2GRAY)
    ftr1 = detector(grabbed)

    while True:
        retval, frame = source.read()
        if not retval:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ftr2, status, err = cv2.calcOpticalFlowPyrLK(grabbed, frame_gray, ftr1)
        good_new = ftr2[status == 1]
        for p in good_new:
            x, y = p.ravel()
            cv2.circle(frame, (x, y), 4, (0, 100, 0), -1)
        writer.write(frame)
        grabbed = frame_gray
        ftr1 = good_new.reshape(-1, 1, 2)

    source.release()
    writer.release()

def main():
    start(harris, "harris_out")
    start(fast, "fast_out")


if __name__ == "__main__":
    main()