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
rootDir = '../testData/fastCamCaps/'
#rootDir = '../testData/756/'
fileList = os.listdir(rootDir)
path = fileList[5]
print(path)

# video capture, takes file path as arg
# integer value for integrated webcam / usb cameras
vidcap = cv2.VideoCapture(rootDir + path)

#vidcap = cv2.VideoCapture(1)
#vidcap.set(cv2.CAP_PROP_FPS, framerate)

# function for reading a frame
# returns boolean for succes/fail and the frame as an ndarray

# create windows
cv2.namedWindow('Circle Detection')
cv2.namedWindow('Difference')

# initialize capture deque
captures = deque()

# list of previous circles
prevCircles = []

n = 0

running = False

def capture():

    while running:
        success, frame = vidcap.read()
        if success:
            captures.append(frame)
        else:
            running = False

def processFrame():

    while running:

        if len(captures) > 0:
            frame = captures.popLeft()

            
    
    diff = frame - prevFrame
    diff = cv2.threshold(diff, differenceThresh, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 15)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)
    #circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, 2, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)
    return diff


if __name__ == '__main__':

    running = True

    vidcap.release()
    cv2.destroyAllWindows()
