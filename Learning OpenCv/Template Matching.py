import cv2
import numpy as np

#Attempted Template Matching with Faces of the Same person

img_bgr = cv2.imread("whole.jpg")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

template = cv2.imread('part.jpg', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9

loc = np.where(res >= threshold)

for point in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, point, (point[0]+w, point[1]+h), (0,255,255), 2)

cv2.imshow('detected', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
