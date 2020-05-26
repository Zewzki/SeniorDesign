import cv2
import os
import numpy as np
import time
from time import sleep
import copy

from multiprocessing.pool import ThreadPool
from collections import deque

print(cv2.__version__)
print('Esc - End')

# define framerate and period of framerate
framerate = 236
showTime = int((1 / framerate) * 1000)

# thresholds for circle dection
highThresh = 75
accumThresh = 30

# min and max radius of circle
minRad = 10
maxRad = 20

# minimum distance between mutliple circles allowed
minDist = 1000

# threshold for filtering out noise when subtracting images
differenceThresh = 150

# threshold for filtering out erroneous circles
# circles that appear at an angle too different from the previous pair will be filtered out
angleThresh = 10

# threshold for list length
# if angle list is > threshold, ball has been detected
listThresh = 7

# amount of time to sleep after ball detection
sleepOnSuccess = 4
lastSuccessTime = time.time() - sleepOnSuccess

# To ensure multiples aren't added
circleAdded = False

# red and green colors for circle display
red = (0, 0, 255)
green = (0, 255, 0)

# for easy file access
#rootDir = '../testData/fastCamCaps/'
rootDir = '../testData/756/'
fileList = os.listdir(rootDir)
path = fileList[2]
print(path)

# video capture, takes file path as arg
# integer value for integrated webcam / usb cameras
vidcap = cv2.VideoCapture(rootDir + path)

#vidcap = cv2.VideoCapture(1)
#vidcap.set(cv2.CAP_PROP_FPS, framerate)

# function for reading a frame
# returns boolean for succes/fail and the frame as an ndarray
success, prevFrame = vidcap.read()

# create windows
cv2.namedWindow('Blob Detection')
cv2.namedWindow('Difference')

# ----

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 70
params.maxThreshold = 90

params.filterByArea = True
params.minArea = 314
params.maxArea = 1256

params.filterByCircularity = True
params.minCircularity = 0.4

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.maxInertiaRatio = .8
params.minInertiaRatio = .2

params.filterByColor = False

# ----

detector = cv2.SimpleBlobDetector_create(params)

# main loop of algorithm
while success:

    # read in frame
    success, frame = vidcap.read()
    
    # exit on fail
    if success == False:
        break

    #diff = cv2.medianBlur(frame, 3) - cv2.medianBlur(prevFrame, 3)
    diff = frame - prevFrame

    # deep copy of frame
    # 'prevFrame = frame' actually saves frame's address
    prevFrame = copy.copy(frame)

    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # perform preprocessing
    ret, diff = cv2.threshold(diff, differenceThresh, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 15)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)
    
    # perform circle detection
    blobs = detector.detect(diff)

    for blob in blobs:
        print(blob.pt)

    diffWithBlobs = cv2.drawKeypoints(diff, blobs, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frameWithBlobs = cv2.drawKeypoints(frame, blobs, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # show iamges to windows
    cv2.imshow('Blob Detection', frameWithBlobs)
    cv2.imshow('Difference', diffWithBlobs)
    
    #key = cv2.waitKey(0)
    key = cv2.waitKey(showTime)
    if key == 27 or not success:
        break

# destroy windows and release file/camera handle
# important, don't remove
cv2.destroyAllWindows()
vidcap.release()
print('done')
