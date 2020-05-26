import os
import cv2
import time

recordTime = 5

framerate = 250;
width = 480
height = 270

vc = cv2.VideoCapture(1)
vc.set(cv2.CAP_PROP_FPS, framerate)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print(vc.get(cv2.CAP_PROP_FPS))
print(vc.get(cv2.CAP_PROP_FRAME_WIDTH), ',', vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not vc.isOpened():
    print('Error opening stream')

counter = 0

times = []

framesList = []

startTime = time.time()

while vc.isOpened() and ((time.time() - startTime) < recordTime):

    #loopstart = time.time()

    success, frame = vc.read()

    if success:

        #frame = cv2.resize(frame, (width, height))
        #out.write(frame)
        #cv2.imshow('Video', frame)
        framesList.append(frame)
        counter = counter + 1

        #key = cv2.waitkey(1)
        #if key == 27:
        #    break

    else:
        break

fps = str(int(counter / (time.time() - startTime)))

path = '../testData/diamondData/'
fileName = str(len(os.listdir(path))) + '-' + fps + '(' + str(width) + ',' + str(height) + ')'
newVidPath = path + str(fileName) + '.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(newVidPath, fourcc, framerate, (width, height))
for frame in framesList:
    frame = cv2.resize(frame, (width, height))
    out.write(frame)

print(time.time() - startTime)
print(counter)
print("FPS: ", fps)

del framesList
vc.release()
out.release()
cv2.destroyAllWindows()
