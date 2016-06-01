import cv2
import numpy as np
#Learning about blurring and smoothing

img = cv2.imread("face.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# hsv hue-sat-value
lower_brown = np.array([20,0,0])
upper_brown = np.array([255,255,255])

mask = cv2.inRange(hsv, lower_brown, upper_brown)
res = cv2.bitwise_and(img, img, mask=mask)
                      
#Smoothing
kernel = np.ones((15,15), np.float32)/225
smoothed = cv2.filter2D(res, -1, kernel)
#Gaussian Blur
blur = cv2.GaussianBlur(res, (15,15), 0)
#Median Blur. Seems like the least noise
median = cv2.medianBlur(res, 15)
#Bilateral Blur
bilateral = cv2.bilateralFilter(res, 15,75,75)

cv2.imshow('bilateral', bilateral)
cv2.imshow("blur", blur)
cv2.imshow("res", res)
cv2.imshow("smoothed", smoothed)
cv2.imshow("median", median)
cv2.waitKey(0)
cv2.destroyAllWindows()
