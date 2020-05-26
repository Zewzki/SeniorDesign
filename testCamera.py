import cv2

vidcap = cv2.VideoCapture(1)
vidcap.set(cv2.CAP_PROP_FPS, 240)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

success, prevFrame = vidcap.read()

cv2.namedWindow('Camera')

while success:

    success, frame = vidcap.read()

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(4)
    if key == 27 or not success:
        break

cv2.destroyAllWindows()
vidcap.release()
