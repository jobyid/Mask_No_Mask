import cv2
import time

video  = cv2.VideoCapture(0)
window = "Capture"
cv2.namedWindow(window)

i = 0

while True:

    ok, frame = video.read()
    if ok:
        cv2.imshow(window,frame)
        if i % 50 == 0:
            name = "imgs/waymoJ" + str(time.time()) + ".png"
            cv2.imwrite(name, frame)

    if cv2.waitKey(1) == ord("q"):
        break
