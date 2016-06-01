# -*- coding: cp1252 -*-
from facerec.feature import Fisherfaces
from facerec.feature import PCA
#from facerec.feature import LDA Too big to use
from facerec.feature import Identity
from facerec.classifier import NearestNeighbor
from facerec.preprocessing import TanTriggsPreprocessing
from facerec.operators import ChainOperator
from facerec.model import PredictableModel
from PIL import Image
import numpy as np
from PIL import Image
import sys, os
import time
#sys.path.append("../..")
import cv2
import multiprocessing


##feature = ChainOperator(TanTriggsPreprocessing(), Fisherfaces())
##classifier = NearestNeighbor()
##model = PredictableModel(feature, classifier)
model = PredictableModel(Fisherfaces(), NearestNeighbor())

img =cv2.imread('group.jpg')
#Choosing the haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#reads the database of faces
def read_images(path, sz=(256,256)):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 
    Returns:
        A list [X,y]
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    folder_names = []
    
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]

def read_imgs(path, sz=(256,256)):
    c = 0
    X,y = [], []
    folder_names = []
    for subdir, dirs, files in os.walk(path):
        for eachFile in files:
            filepath = subdir + os.sep + eachFile
            try:
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                # resize to given size (if given)
                if (sz is not None):
                    im = cv2.resize(im, sz)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
            except IOError, (errno, strerror):
                print "I/O error({0}): {1}".format(errno, strerror)
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
        c = c+1
    return [X,y]

#Directory of the face database
pathdir='C:\Users\Nathan Jiang\Desktop\sort\dataset'


#Initialization: Finds the desired image
##print "enter which member u want to detect"
##name = raw_input().strip()
##count = 0
##if not os.path.exists(os.path.join(pathdir+name)):
##    os.makedirs(os.path.join(pathdir+name))
##    print 'No path'
    
######################################
#Going through the database
[X,y,subject_names] = read_images(pathdir)

#[X,y] = read_imgs('C:\Users\Nathan Jiang\Desktop\sort\dataset\Tzuyu')
#subject_names = [name]

#Creates a list of the number of members
list_of_labels = list(xrange(max(y)+1))
#Maps a dictionary between the numbers and the names of the individuals
subject_dictionary = dict(zip(list_of_labels, subject_names))
model.compute(X,y)

#Finds the faces

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces = face_cascade.detectMultiScale(gray, 1.2, 1)

for (x,y,w,h) in faces:
    #Draws the rectangle
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #Basically
    resized_image = cv2.resize(img[y:y+h,x:x+w], (273, 273))
    
    #Same things except this one is gray
    sampleImage = gray[y:y+h, x:x+w]
    sampleImage = cv2.resize(sampleImage, (256,256))
    
    #Facial Recognition
    [predicted_label, generic_classifier_output] = model.predict(sampleImage)
    print [ predicted_label, generic_classifier_output]

    #Tzuyu = 1, chaeyoung = 0
    #Choosing a similarity threshold. The higher the threshold, the less accuracy
    #This is for PCA
    #if int(generic_classifier_output['distances']) <=  10000:

    #This is for FisherFaces
    if int(generic_classifier_output['distances']) <=  800:
        #cv2.putText(img,str(generic_classifier_output), (x,y), cv2.FONT_HERSHEY_PLAIN,0.8, (255,255,255),1)
        cv2.putText(img,str(subject_dictionary[predicted_label]), (x,y), cv2.FONT_HERSHEY_PLAIN,0.8, (255,255,255),1)
    #cv2.imwrite( str(count)+'.jpg', resized_image );
    #count += 1
cv2.imshow('Recognition',img)

    
cv2.waitKey(0)
cv2.destroyAllWindows()
######################################################






#comincia il riconoscimento.
