import cv2
import numpy as np
from time import sleep
print(cv2.__version__)

highThresh = 50
accumThresh = 40
minRad = 20
maxRad = 30
minDist = 10

red = (0, 0, 255)
green = (0, 255, 0)

vidcap = cv2.VideoCapture('../testData/756/isolatedPitches.mp4')

success, im = vidcap.read()

cv2.namedWindow('Circle Detection')

while success:
    
    success, frame = vidcap.read()
    #print('frame captured: ', success)
    colorFrame = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 5)
    #colorFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    try:

        #t = time.time()
        
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, minDist, param1 = highThresh, param2 = accumThresh, minRadius = minRad, maxRadius = maxRad)

        #deltaTime = time.time() - t
        #print(deltaTime)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(colorFrame, (i[0], i[1]), i[2], green, 2)
                cv2.circle(colorFrame, (i[0], i[1]), 2, red, 3)

        cv2.imshow('Circle Detection', colorFrame)

    except:
        #cv2.imshow('Circle Detection', colorFrame)
        cv2.destroyWindow('Circle Detection')
        vidcap.release()

    key = cv2.waitKey(20)
    if key == 27 or not success:
        cv2.destroyWindow('Circle Detection')
        vidcap.release()
        break
