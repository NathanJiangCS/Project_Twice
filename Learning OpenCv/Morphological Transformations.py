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


#Erosion Transformation

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(mask, kernel, iterations=1)
#Dilation Transformation
dilation = cv2.dilate(mask, kernel, iterations=1)

#opening to remove false positives
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#closing to remove false negatives
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('img', img)
cv2.imshow("res", res)
#cv2.imshow("erosion", erosion)
#cv2.imshow("dilation", dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
