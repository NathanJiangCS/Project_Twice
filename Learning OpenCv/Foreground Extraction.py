import cv2
import numpy as np
import matplotlib.pyplot as plt
#The result of this code is the isolation of the foreground image within the rect
#specified. Using grabCut, it takes out what is considered the foreground
#and the rest of the image becomes a black background


img = cv2.imread("whole.jpg")
mask = np.zeros(img.shape[:2], np.uint8)


#Background Model and Foreground Model
bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

#rectangle encompassing what we consider as the foreground of the image
rect = (430, 130, 300, 800)


cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()
