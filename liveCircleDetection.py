import numpy as np
import cv2
import time

highThresh = 50
accumThresh = 30
minRad = 10
maxRad = 50
minDist = 20

red = (0, 0, 255)
green = (0, 255, 0)

cv2.namedWindow('Circle Detection')
vc = cv2.VideoCapture(1)
im = cv2.imread('pitch.jpg', 1)

while True:

    success, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 5)
    colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    try:

        #t = time.time()
        
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)
        circles = np.uint16(np.around(circles))

        #deltaTime = time.time() - t
        #print(deltaTime)

        for i in circles[0, :]:
            cv2.circle(colorFrame, (i[0], i[1]), i[2], green, 2)
            cv2.circle(colorFrame, (i[0], i[1]), 2, red, 3)

        cv2.imshow('Circle Detection', colorFrame)

    except:
        cv2.imshow('Circle Detection', colorFrame)
        #cv2.destroyWindow('Circle Detection')
        #vc.release()

    key = cv2.waitKey(20)
    if key == 27 or not success:
        cv2.destroyWindow('Circle Detection')
        vc.release()
        break
