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
rootDir = '../testData/diamondData/zone1/'
fileList = os.listdir(rootDir)
path = fileList[0]
print(path)
vidcap = cv2.VideoCapture(rootDir + path)

# list of previous balls
b0 = []
n = 0
success, prevFrame = vidcap.read()
trajectoryLine = np.zeros_like(prevFrame)
# Create a mask image for drawing purposes
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# main loop of algorithm
while(success):
    # read in frame
    success, frame = vidcap.read()
    trajectoryFrame = copy.copy(frame)

    # exit on fail
    if success == False:
        break

    mask = subtractor.apply(frame)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)
    
    key = cv2.waitKey(showTime)
    if circles is not None:
        
        if circles[0][0][2] != 0:
            n = n + 1
        try:
            convertedCircles = np.uint16(np.around(circles))
        except:
            continue

        # loop through all found circles
        for i in convertedCircles:

            i = i[0]
            
            # (i[0], i[1]) is (x, y); i[2] is radius (radius green circle)
            cv2.circle(frame, (i[0], i[1]), i[2], green, 2)
            # (i[0], i[1]) is (x, y); 2 is radius (center red dot)
            cv2.circle(frame, (i[0], i[1]), 2, red, 3)

            # same but circling on diff
            cv2.circle(mask, (i[0], i[1]), i[2], green, 2)
            cv2.circle(mask, (i[0], i[1]), 2, red, 3)


        key = cv2.waitKey(0)

    cv2.imshow('Circle Detection', frame)
    cv2.imshow("Background Subtraction", mask)

    if key == 27 or not success:
        break

print(n)
# destroy windows and release file/camera handle
# important, don't remove
cv2.destroyAllWindows()
vidcap.release()
print('done')
