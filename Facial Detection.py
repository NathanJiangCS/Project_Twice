import cv2
import numpy as np
import sys
import urllib
#Cascade Classifiers from https://github.com/Itseez/opencv/tree/master/data/haarcascades
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
db = open('links.txt','r')
count = 420
for i in db:
    i= i.strip()
    if i[:4] == 'http':
        req = urllib.urlopen(i)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

        img = cv2.imdecode(arr, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]

            roi_color = img[y-30:y+h+30, x-30:x+w+30]
            cv2.imwrite('%s.png' %count, roi_gray)
            count += 1
                        
##            cv2.imshow('img',img)
##            cv2.imshow('roi_gray',roi_gray)
##            
##
##            cv2.waitKey(0)
##            cv2.destroyAllWindows()
        ##    eyes = eye_cascade.detectMultiScale(roi_gray)
        ##    for (ex, ey, ew, eh) in eyes:
        ##        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)



        
