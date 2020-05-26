
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

# define camera width and height
sWidth = 640
sHeight = 360

# thresholds for circle dection
highThresh = 75
accumThresh = 30

# min and max radius of circle
minRad = 2
maxRad = 15

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
blue = (255, 0, 255)
white = (255, 255, 255)

# for easy file access
#rootDir = '../testData/fastCamCaps/'
#rootDir = '../testData/756/'
rootDir = '../testData/diamondData/zone3/'
fileList = os.listdir(rootDir)
path = fileList[2]
print(path)

#decisionLine = sHeight - 300
decisionLine = sHeight / 2
slopeThresh = sWidth / sHeight

def evaluateZone(circleList):

    if len(circleList) <= 0:
        print('circle length <= %d' % (len(circleList)))
        return

    sumSlope = 0
    sumRad = 0
    sumX = 0
    sumY = 0
    zeroDivs = 0

    for i in range(0, len(circleList) - 1):
        
        c1 = circleList[i][0]
        c2 = circleList[i + 1][0]

        x1, y1 = c1[1], c1[0]
        x2, y2 = c2[1], c2[0]

        if x2 - x1 == 0:
            continue

        try:

            slope = ((y2 - y1) / (x2 - x1))

            if slope < slopeThresh or slope > -slopeThresh:

                sumSlope += slope

                sumX += x1
                sumY += y1
                sumRad += c1[2]
            
        except ZeroDivisionError:
            zeroDivs += 1

    sumRad += circleList[len(circleList) - 1][0][2]
    sumX += circleList[len(circleList) - 1][0][1]
    sumY += circleList[len(circleList) - 1][0][0]

    avgRad = int(sumRad / len(circleList))
    avgSlope = sumSlope / (len(circleList) - zeroDivs)
    avgX = sumX / len(circleList)
    avgY = sumY / len(circleList)

    first = circleList[0][0]
    last = circleList[len(circleList) - 1][0]

    #avgSlope = (last[2] - first[2]) / (last[1] - first[1])
    print('slope: %f' % avgSlope)
    
    #intercept = (avgX + (avgY * avgSlope))
    intercept = -(avgX * avgSlope) + avgY
    
    print(avgSlope)
    print(avgRad)

    zoneList = []

    midPoint = sWidth / 2
    
    for i in range(-3, 4):
        zoneList.append([midPoint - avgRad + (i * (2 * avgRad)), midPoint + avgRad + (i * (2 * avgRad))])

    posAtDecisionLine = (intercept + (avgSlope * decisionLine))

    print('Position at Line: %f' % (posAtDecisionLine))

    zone = 0

    print(zoneList)
    
    for i in range(0, len(zoneList)):
        leftBound = zoneList[i][0]
        rightBound = zoneList[i][1]
        if posAtDecisionLine > leftBound and posAtDecisionLine < rightBound:
            # add 1 to offset index
            zone = i + 1
            break

    return (avgSlope, intercept), zone, zoneList



# video capture, takes file path as arg
# integer value for integrated webcam / usb cameras
vidcap = cv2.VideoCapture(rootDir + path)

#vidcap = cv2.VideoCapture(0)
#vidcap.set(cv2.CAP_PROP_FPS, framerate)
#vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, sWidth)
#vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, sHeight)

subtractor = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 50, detectShadows = False)

def updateBackground():

    for i in range(0, 100):
        
        success, f = vidcap.read()
        
        if not success:
            continue
        
        subtractor.apply(f)

# function for reading a frame
# returns boolean for succes/fail and the frame as an ndarray
success, prevFrame = vidcap.read()
sWidth = prevFrame.shape[1]
sHeight = prevFrame.shape[0]

print(sWidth, sHeight)

# create windows
cv2.namedWindow('Circle Detection')
cv2.namedWindow('Difference')

# list of previous circles
prevCircles = []

n = 0

framesWithoutCircleThresh = 30
framesSinceLastCircle = 0

ballDetectedLastFrame = False

# main loop of algorithm
while success:

    # read in frame
    success, frame = vidcap.read()
    
    # exit on fail
    if success == False:
        break
    
    diff = subtractor.apply(frame)

    ballDetected = False

    if (time.time() - lastSuccessTime) > sleepOnSuccess:
    
        # perform circle detection
        circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, 2, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)

        # perform algorithm
        if circles is not None:
            prevCircles.append(circles[0])
            framesSinceLastCircle = 0
        else:
            framesSinceLastCircle += 1

        if framesSinceLastCircle > framesWithoutCircleThresh:

            if len(prevCircles) > 4:

                (slope, intercept), zone, zoneList = evaluateZone(prevCircles)
                lastSuccessTime = time.time()
                prevCircles.clear()

                #slope *= -1
                slope = slope / 1

                cv2.line(frame, (int(intercept), 0), (int(intercept + (slope * sHeight)), sHeight), green, 3)

                for z in zoneList:
                    cv2.line(frame, (int(z[0]), 0), (int(z[0]), sHeight), red, 1)
                    cv2.line(frame, (int(z[1]), 0), (int(z[1]), sHeight), red, 1)

                cv2.line(frame, (0, int(decisionLine)), (sWidth, int(decisionLine)), blue, 2)

                #cv2.line(frame, (int(sWidth / 2), 0), (int(sWidth / 2), sHeight), white, 2)
            
                cv2.imshow('Circle Detection', frame)

                print('Zone: %s' % (zone))
            
                key = cv2.waitKey(0)

                updateBackground()
            
            prevCircles.clear()

        angles = []

        if len(prevCircles) > 1:

            for i in range(1, len(prevCircles)):

                #c1 = prevCircles[0][0]
                c1 = prevCircles[i-1][0]
                c2 = prevCircles[i][0]

                xDist = c1[0] - c2[0]
                yDist = c1[1] - c2[1]

                if not(xDist == 0):
                    theta = np.degrees(np.arctan(yDist / xDist))
                    angles.append(theta)

        if len(prevCircles) > 2:

            ballDetected = True
        
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


    ballDetectedLastFrame = ballDetected
    
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
