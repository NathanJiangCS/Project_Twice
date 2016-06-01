import cv2
import numpy as np

#Testing the Haar Cascade
#Cascade Classifiers from https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#cap = cv2.VideoCapture(0)
img = cv2.imread('group.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 1)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]

    roi_color = img[y-30:y+h+30, x-30:x+w+30]
##    eyes = eye_cascade.detectMultiScale(roi_gray)
##    for (ex, ey, ew, eh) in eyes:
##        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)



cv2.imshow('img',img)


cv2.waitKey(0)
cv2.destroyAllWindows()
