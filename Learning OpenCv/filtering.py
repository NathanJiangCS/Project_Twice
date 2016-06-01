import cv2
import numpy as np

img = cv2.imread("face.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# hsv hue-sat-value
lower_brown = np.array([20,0,0])
upper_brown = np.array([255,255,255])

mask = cv2.inRange(hsv, lower_brown, upper_brown)
res = cv2.bitwise_and(img, img, mask=mask)


cv2.imshow('img', img)
cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
