#!/usr/bin/env python
import numpy as np
import cv2

TRACK_POINTS_NUM = 100
HARRIS_QUALITY_LEVEL = 0.13
HARRIS_MIN_DISTANCE = 5
HARRIS_BLOCK_SIZE = 5

def harris(frame):
    return cv2.goodFeaturesToTrack(frame,
                                   mask=None,
                                   useHarrisDetector=True,
                                   maxCorners=TRACK_POINTS_NUM,
                                   qualityLevel=HARRIS_QUALITY_LEVEL,
                                   minDistance=HARRIS_MIN_DISTANCE,
                                   blockSize=HARRIS_BLOCK_SIZE)

def fast(frame):
    fast = cv2.FastFeatureDetector()
    f_features = list(sorted(fast.detect(frame, None), key=lambda l: -l.response))[: TRACK_POINTS_NUM]

    return np.array([[f.pt] for f in f_features], np.float32)
    
    
def open_videos(in_filename, out_filename):
    source_video = cv2.VideoCapture(in_filename)
    width = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(source_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(source_video.get(cv2.cv.CV_CAP_PROP_FPS))
    fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')
    return source_video, cv2.VideoWriter(out_filename, fourcc=fourcc, fps=fps, frameSize=(width, height))
    

def start(detector, out_filename):
    source , writer = open_videos('sequence.mpg', '%s.mpg' % out_filename)
    
    retval, grabbed = source.read()
    out_img = np.zeros_like(grabbed)
    colors = np.random.randint(0, 255, (TRACK_POINTS_NUM, 3))
    
    if not retval:
        return
    grabbed = cv2.cvtColor(grabbed, cv2.COLOR_BGR2GRAY)
    ftr1 = detector(grabbed)
    
   

    while True:
        retval, frame = source.read()
        if not retval:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ftr2, status, _ = cv2.calcOpticalFlowPyrLK(grabbed, frame_gray, ftr1, None,
                                                            winSize=(15, 15),
                                                            maxLevel=2,
                                                            criteria=(
                                                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                10,
                                                                0.03))
                                                                
                                                                
        prev_points, next_points = ftr1[status == 1], ftr2[status == 1]

        for i, (next, prev) in enumerate(zip(next_points, prev_points)):
            a, b = next.ravel()
            c, d = prev.ravel()
            cv2.line(out_img, (a, b), (c, d), colors[i].tolist(), 1)
            cv2.circle(frame, (a, b), 2, colors[i].tolist(), 2)

        output_frame = cv2.add(frame, out_img)
        writer.write(output_frame)

        grabbed = frame_gray.copy()
        ftr1 = next_points.reshape(-1, 1, 2)
    

    source.release()
    writer.release()

def main():
    start(harris, "harris_out")
    start(fast, "fast_out")


if __name__ == "__main__":
    main()