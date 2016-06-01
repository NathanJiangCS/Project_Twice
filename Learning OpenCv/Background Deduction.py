import cv2
import numpy as np

#Using motion to reduce background and single out the focus
#uses webcam, somewhat works

cap = cv2.VideoCapture(0)

fgbg = cv2.BackgroundSubtractorMOG()
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow("original",frame)
    cv2.imshow("fg",fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
