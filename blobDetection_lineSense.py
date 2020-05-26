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

# line detector variables #############
beginFC = False
xCords = []
yCords = []
fC = 0
slopeTol = 0.005
intrcptTol = 5
maxSlope = 0.8
#maxSlope = 1 / maxSlope
###################################
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
#rootDir = '../testData/756/'
rootDir = '../testData/fastCamCaps/'
fileList = os.listdir(rootDir)
# works with list 5,6,9,10
path = fileList[6]
print(path)

# video capture, takes file path as arg
# integer value for integrated webcam / usb cameras
vidcap = cv2.VideoCapture(rootDir + path)

#vidcap = cv2.VideoCapture(0)
#vidcap.set(cv2.CAP_PROP_FPS, framerate)

# function for reading a frame
# returns boolean for succes/fail and the frame as an ndarray
success, prevFrame = vidcap.read()

# create windows
cv2.namedWindow('Blob Detection')
cv2.namedWindow('Difference')

#linear pattern recognition
def circleInLine(xCords,yCords):
    slopeArr = []
    intrcptArr = []
    sccss = 0
    ss =[]
    si = []
    i=0
    b=0

    for b in range(0,len(xCords)):
        for i in range(b, len(xCords)):
            if i != b:
                slope = (yCords[b]-yCords[i])/(xCords[b]-xCords[i])
                intrcpt = slope*xCords[b]+yCords[b]
                if slope < maxSlope and slope > -maxSlope:
                   slopeArr.append(slope)
                   intrcptArr.append(intrcpt)
    if len(slopeArr) > 3:
        a = 0
        c = 0
        for a in range(0,len(slopeArr)):
            for c in range(0,len(slopeArr)):
                if a!=c:
                    if(slopeArr[a]+slopeTol) > slopeArr[c] and  (slopeArr[a]-slopeTol) < slopeArr[c]:
                        sccss+=1
                        ss.append(slopeArr[c])
                        si.append(intrcptArr[c])
                        if sccss > 2:
                            print('------------------')
                            print('lineFound')
                            print('xcord: ',xCords)
                            print('ycord: ',yCords)
                            print('slopes: ',slopeArr)
                            print('ss : ', ss)
                            print('intercepts: ',intrcptArr)
                            print('------------------')
                            return ss, si
            sccss = 0
    return [], []
# ----

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 70
params.maxThreshold = 90

params.filterByArea = True
params.minArea = 400
params.maxArea = 3000

params.filterByCircularity = True
params.minCircularity = 0.8
#params.maxCircularity = 0.97

params.filterByConvexity = True
params.minConvexity = 0.7

params.filterByInertia = False
#params.maxInertiaRatio = .8
# was 0.6
params.minInertiaRatio = .5

params.filterByColor = False

# ----

detector = cv2.SimpleBlobDetector_create(params)
count = 0
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
    #prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # perform preprocessing
    ret, diff = cv2.threshold(diff, differenceThresh, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 33)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)

    #ret, frame = cv2.threshold(frame, differenceThresh, 255, cv2.THRESH_TOZERO)
    #ret, prevFrame = cv2.threshold(prevFrame, differenceThresh, 255, cv2.THRESH_TOZERO)

    #frame = cv2.medianBlur(frame, 3)
    #prevFrame = cv2.medianBlur(prevFrame, 5)

    #frame = cv2.GaussianBlur(frame, (9,9),0)
    #prevFrame = cv2.GaussianBlur(prevFrame, (9,9),0)


    
    #diff = frame - prevFrame

    
    # perform circle detection
    blobs = detector.detect(diff)

    for blob in blobs:
        #print(blob.pt)
        beginFC = True
        xCords.append(blob.pt[0])
        yCords.append(blob.pt[1])
    if fC > 9 and len(xCords) > 2:

        slopes = []
        intercepts = []

        slopes, intercepts = circleInLine(xCords,yCords)

        for j in range(0, len(slopes)):
            
            s, i, w = slopes[j], intercepts[j], frame.shape[1]
            cv2.line(diff, (0, int(i)), (w, int(i + (s * w))), (255, 0, 0), 2)
            cv2.line(frame, (0, int(i)), (w, int(i + (s * w))), (255, 0, 0), 2)

            cv2.imshow('Blob Detection', frame)
            cv2.imshow('Difference', diff)

            key = cv2.waitKey(0)
            if key == 27 or not success:
                break
            
    if beginFC == True:
        fC +=1
        if fC > 10:
            fC = 0
            beginFC = False
            xCords.clear()
            yCords.clear()
    diffWithBlobs = cv2.drawKeypoints(diff, blobs, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frameWithBlobs = cv2.drawKeypoints(frame, blobs, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # show iamges to windows
    cv2.imshow('Blob Detection', frameWithBlobs)
    cv2.imshow('Difference', diffWithBlobs)

    key = cv2.waitKey(showTime)
    
    #if count > 867:
    #    key = cv2.waitKey(0)
    #else:
    #    key = cv2.waitKey(showTime)
    if key == 27 or not success:
        break
    count+=1
# destroy windows and release file/camera handle
# important, don't remove
cv2.destroyAllWindows()
vidcap.release()
print('done')
