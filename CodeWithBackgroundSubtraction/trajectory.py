import cv2
import os
import numpy as np
import time
from time import sleep
import copy

print(cv2.__version__)
print('Esc - End')
# define framerate and period of framerate
framerate = 236
showTime = int((1 / framerate) * 1000 )
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
slowTime = 1000
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
blue = (255, 0, 0)
# for easy file access
rootDir = '../testData/diamondData/zone3/'
fileList = os.listdir(rootDir)
path = fileList[0]
print(path)
vidcap = cv2.VideoCapture(rootDir + path)

n = 0
ball0 = []
success, prevFrame = vidcap.read()
trajectoryLine = np.zeros_like(prevFrame)
# Create a mask image for drawing purposes
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)


# main loop of algorithm
while(success):
    # read in frame
    success, frame = vidcap.read()
    # exit on fail
    if success == False:
        break

    mask = subtractor.apply(frame)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)
    
    key = cv2.waitKey(showTime)
    if circles is not None:
        print(circles[0][0][2])

        if len(ball0) < 1:
            ball0 = circles[0]
        elif circles[0][0][2] > 5:
            ball1 = circles[0]
            trajectoryLine = cv2.line(trajectoryLine, (ball1[0][0], ball1[0][1]),(ball0[0][0], ball0[0][1]), blue, 2)
            frame = cv2.circle(frame, (ball1[0][0], ball1[0][1]), 5, blue, -1)
            ball0 = ball1
        key = cv2.waitKey(0)
    frame = cv2.add(frame, trajectoryLine)
    cv2.imshow("Trajectory", frame)

    if key == 27 or not success:
        break

# destroy windows and release file/camera handle
# important, don't remove
cv2.destroyAllWindows()
vidcap.release()
print('done')
