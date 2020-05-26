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
drawZonesAndCircle = True
# red and green colors for circle display
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
# for easy file access
rootDir = '../testData/diamondData/zone5/'
fileList = os.listdir(rootDir)
path = fileList[1]
print(path)
vidcap = cv2.VideoCapture(rootDir + path)

# list of previous balls
b0 = []
n = 0
success, prevFrame = vidcap.read()
zoneFrame = np.zeros_like(prevFrame)
# Create a mask image for drawing purposes
subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

n = 0
xAvg = 0
yAvg = 0
rAvg = 0
# list of zones
zone = [[] for i in range(7)]
# list of previous circles
prevCircles = []
xMax = 480

def drawZones(frame, radius):
    x = 240
    y1 = 0
    y2 = 270
    diam = 2 * radius

    cv2.line(frame, (int(x + radius), y1),(int(x + radius), y2), blue, 1)
    cv2.line(frame, (int(x - radius), y1),(int(x - radius), y2), blue, 1)
    for i in range(1, 4):
        cv2.line(frame, (int(x + radius + i*diam), y1),(int(x + radius + i*diam), y2), blue, 1)
        cv2.line(frame, (int(x - radius - i*diam), y1),(int(x - radius - i*diam), y2), blue, 1)

    convertedCircles = np.uint16(np.around(prevCircles))
    # loop through all found circles
    for i in convertedCircles:
        i = i[0]
        # (i[0], i[1]) is (x, y); i[2] is radius (radius green circle)
        cv2.circle(frame, (i[0], i[1]), int(radius), green, 2)
        # (i[0], i[1]) is (x, y); 2 is radius (center red dot)
        cv2.circle(frame, (i[0], i[1]), 2, red, 3)
        if (i[0] > (240 - 2 * radius - 3 * diam)) & (i[0] < (240 + 2 * radius + 3 * diam)):
            if i[0] < (240 - radius - 2 * diam):
                zone[0].append(i)
            elif i[0] < (240 - radius - diam):
                zone[1].append(i)
            elif i[0] < (240 - radius):
                zone[2].append(i)
            elif i[0] < (240 + radius):
                zone[3].append(i)
            elif i[0] < (240 + radius + diam):
                zone[4].append(i)
            elif i[0] < (240 + radius + 2 * diam):
                zone[5].append(i)
            else:
                zone[6].append(i)





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
        n = n + 1
        xAvg = xAvg + circles[0][0][0]
        yAvg = yAvg + circles[0][0][1]
        rAvg = rAvg + circles[0][0][2]
        nextCircle = circles[0]
        if (nextCircle[0][0] > 156) & (nextCircle[0][0] < 324):
            prevCircles.append(nextCircle)

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

    
    cv2.imshow('Circle Detection', frame)
    cv2.imshow("Background Subtraction", mask)

    if key == 27 or not success:
        break
        


xAvg = xAvg/n
yAvg = yAvg/n
rAvg = rAvg/n




print('n = ', n)
print('xAvg = ', xAvg)
print('yAvg = ', yAvg)
print('rAvg = ', rAvg)

drawZones(zoneFrame, rAvg)
for i in range(0, len(zone)):
    print('Zone ', i + 1, ' count : ', len(zone[i]))

while(True):
    cv2.imshow('Zones', zoneFrame)
    key = cv2.waitKey(0)
    if key == 27:
        break
# destroy windows and release file/camera handle
# important, don't remove
cv2.destroyAllWindows()
vidcap.release()
print('done')
