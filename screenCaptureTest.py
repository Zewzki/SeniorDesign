import numpy as np
import cv2
from mss import mss

highThresh = 50
accumThresh = 30
minRad = 10
maxRad = 20
minDist = 20

red = (0, 0, 255)
green = (0, 255, 0)
boundingBox = {'top': 100, 'left': 40, 'width':800, 'height': 800}

cv2.namedWindow('Circle Detection')

sct = mss()

while True:

    cap = sct.grab(boundingBox)

    cap = np.array(cap)
    
    grayCap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    grayCap = cv2.medianBlur(grayCap, 5)

    circles = cv2.HoughCircles(grayCap, cv2.HOUGH_GRADIENT, 1, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(cap, (i[0], i[1]), i[2], green, 2)
            cv2.circle(cap, (i[0], i[1]), 2, red, 3)

    cv2.imshow('Circle Detection', cap)
    
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyWindow('screenCap')
        break
