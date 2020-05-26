
#import subprocess
#import sys

import cv2
import os
import numpy as np
import time
from time import sleep
import copy

#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])

print(cv2.__version__)
print('Esc - End')

# define framerate and period of framerate
framerate = 236
showTime = int((1 / framerate) * 1000)

# thresholds for circle dection
highThresh = 75
accumThresh = 30

# min and max radius of circle
minRad = 5
maxRad = 10

# minimum distance between mutliple circles allowed
minDist = 1000

# threshold for filtering out noise when subtracting images
differenceThresh = 150

# threshold for filtering out erroneous circles
# circles that appear at an angle too different from the previous pair will be filtered out
angleThresh = 10

# threshold for list length
# if angle list is > threshold, ball has been detected
listThresh = 4

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
#rootDir = '../testData/756/'
rootDir = '../testData/diamondData/zone4/'
fileList = os.listdir(rootDir)
path = fileList[0]
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
cv2.namedWindow('Circle Detection')
cv2.namedWindow('Difference')

# list of previous circles
prevCircles = []

n = 0

subtractor = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 50, detectShadows = False)

# main loop of algorithm
while success:

    # read in frame
    success, frame = vidcap.read()
    
    # exit on fail
    if success == False:
        break

    #diff = cv2.medianBlur(frame, 3) - cv2.medianBlur(prevFrame, 3)
    #diff = frame - prevFrame
    diff = subtractor.apply(frame)

    # deep copy of frame
    # 'prevFrame = frame' actually saves frame's address
    prevFrame = copy.copy(frame)

    #diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # perform preprocessing
    #ret, diff = cv2.threshold(diff, differenceThresh, 255, cv2.THRESH_BINARY)
    #diff = cv2.medianBlur(diff, 15)
    #diff = cv2.GaussianBlur(diff, (9, 9), 0)

    ballDetected = False

    if (time.time() - lastSuccessTime) > sleepOnSuccess:
    
        # perform circle detection
        circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, 2, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)

        # perform algorithm
        if circles is not None:
            prevCircles.append(circles[0])
        else:
            if len(prevCircles) > listThresh:
                ballDetected = True
            else:
                prevCircles.clear()

        angles = []

        if len(prevCircles) > 1:

            for i in range(1, len(prevCircles)):

                c1 = prevCircles[0][0]
                c2 = prevCircles[i][0]

                xDist = c1[0] - c2[0]
                yDist = c1[1] - c2[1]

                if not(xDist == 0):
                    theta = np.degrees(np.arctan(yDist / xDist))
                    angles.append(theta)

        if len(angles) > 1:

            threshBroken = False

            angDiff = abs(max(angles) - min(angles))

            if angDiff > angleThresh:
                threshBroken = True

            if threshBroken:

                prevCircles.pop(0)

                if len(prevCircles) > listThresh:

                    ballDetected = True

            else:

                print('Max: ', max(angles))
                print('Min: ', min(angles))
                print('Diff: ', angDiff)
                print()
        
        if ballDetected:
            print(angles)
            lastSuccessTime = time.time()
            prevCircles.clear()
            #print(prevCircles[len(prevCircles) - 1])
                
        # convert diff to color so we can lay green circles on top
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    # draw circles on images
    if prevCircles is not None:
        
        try:
            convertedPrevCircles = np.uint16(np.around(prevCircles))
        except:
            continue

        # loop through all found circles
        for i in convertedPrevCircles:

            i = i[0]
            
            # (i[0], i[1]) is (x, y); i[2] is radius (radius green circle)
            cv2.circle(frame, (i[0], i[1]), i[2], green, 2)
            # (i[0], i[1]) is (x, y); 2 is radius (center red dot)
            cv2.circle(frame, (i[0], i[1]), 2, red, 3)

            # same but circling on diff
            cv2.circle(diff, (i[0], i[1]), i[2], green, 2)
            cv2.circle(diff, (i[0], i[1]), 2, red, 3)

    # show iamges to windows
    cv2.imshow('Circle Detection', frame)
    cv2.imshow('Difference', diff)
    
    #key = cv2.waitKey(0)
    key = cv2.waitKey(showTime)
    if key == 27 or not success:
        break

# destroy windows and release file/camera handle
# important, don't remove
cv2.destroyAllWindows()
vidcap.release()
print('done')
