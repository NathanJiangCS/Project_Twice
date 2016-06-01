# -*- coding: cp1252 -*-
from facerec.feature import Fisherfaces
from facerec.feature import PCA
#from facerec.feature import LDA Too big to use
from facerec.feature import Identity
from facerec.classifier import NearestNeighbor
from facerec.preprocessing import TanTriggsPreprocessing
from facerec.operators import ChainOperator
from facerec.model import PredictableModel
import urllib #Required to open links that we got from the scraper
import numpy as np
from PIL import Image
import sys, os
#sys.path.append("../..")
import cv2
import multiprocessing

##Selecting 3 models to base the facial recognition off of. The results of these 3 will be averaged at the end for a more reliable result

##Alternative Approach are the following 3 Lines:
##feature = ChainOperator(TanTriggsPreprocessing(), Fisherfaces())
##classifier = NearestNeighbor()
##model = PredictableModel(feature, classifier)

model = PredictableModel(Fisherfaces(), NearestNeighbor())
model1 = PredictableModel(PCA(), NearestNeighbor())
model2 = PredictableModel(Identity(), NearestNeighbor())

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

#Creates a list of the number of members
list_of_labels = list(xrange(max(y)+1))
#Maps a dictionary between the numbers and the names of the individuals
subject_dictionary = dict(zip(list_of_labels, subject_names))

#Using the 3 Models to compute Similarities Based on Data Sets
model.compute(X,y)
model1.compute(X,y)
model2.compute(X,y)

######################################
#Loading the Pictures
pictures = open('links.txt','r')
for i in pictures:
    i= i.strip()
    if i[:4] == 'http':
        req = urllib.urlopen(i)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        #Each picture we analyze is stored in img
        img = cv2.imdecode(arr, -1)

        #Now doing facial Detection. For more information, refer to Facial Detection.py. The process is the same as the one done here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        #The last value in the parameters determines the threshold. A higher number will give you more accurate detections but less detections
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)

        FisherFacePred = -1
        PCAPred = -1
        IdentityPred = -1
        #Performing facial recognition based off the face database given
        for (x,y,w,h) in faces:
            
            #Draws the rectangle on image of where face was detected
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
            #Converts to gray image which is easier to interpret
            sampleImage = gray[y:y+h, x:x+w]

            #Resizing to match the database image sizes
            sampleImage = cv2.resize(sampleImage, (256,256))

            #################################
            #Facial Recognition with FisherFaces
            [predicted_label, generic_classifier_output] = model.predict(sampleImage)
            
            #Choosing a similarity threshold. The higher the threshold, the less accuracy
            if int(generic_classifier_output['distances']) <=  800:
                FisherFacePred = predicted_label

                
            #################################
            #Facial Recognition with PCA
            [predicted_label, generic_classifier_output] = model1.predict(sampleImage)

            #Choosing a similarity threshold. The higher the threshold, the less accuracy
            if int(generic_classifier_output['distances']) <=  10000:
                PCAPred = predicted_label

            #################################
            #Facial Recognition with Identity Analysis
            
            [predicted_label, generic_classifier_output] = model2.predict(sampleImage)
            IdentityPred = predicted_label

            #################################
            #Logic to determine which label is chosen
            finalLabel = -1
            print FisherFacePred, PCAPred, IdentityPred
        
            if FisherFacePred != -1 and PCAPred != -1 and IdentityPred != -1: #All non zero values
                if FisherFacePred == PCAPred == IdentityPred:
                    finalLabel = FisherFacePred
                elif FisherFacePred == PCAPred:
                    finalLabel = FisherFacePred
                elif FisherFacePred == IdentityPred:
                    finalLabel = FisherFacePred
                elif PCAPred == IdentityPred:
                    finalLabel = PCAPred

                else: #When none of them are the same, trust FisherFace
                    finalLabel = FisherFacePred
                    
            elif PCAPred == -1 and FisherFacePred != -1 and IdentityPred != -1:
                if FisherFacePred == IdentityPred:
                    finalLabel = FisherFacePred
                else:
                    finalLabel = FisherFacePred
                    
            elif FisherFacePred == -1 and PCAPred != -1 and IdentityPred != -1:
                if PCAPred == IdentityPred:
                    finalLabel = PCAPred
                else:
                    finalLabel = IdentityPred
                    
            elif IdentityPred == -1 and PCAPred != -1 and FisherFacePred != -1:
                if PCAPred == FisherFacePred:
                    finalLabel = FisherFacePred
                else:
                    finalLabel = FisherFacePred
                    
            elif FisherFacePred != -1:
                finalLabel = FisherFacePred
                
            elif PCAPred != -1:
                finalLabel = PCAPred
                
            elif IdentityPred != -1:
                finalLabel = IdentityPred
                
            else:
                finalLabel = -1 #No one knows 
                
                
                #cv2.putText(img,str(generic_classifier_output), (x,y), cv2.FONT_HERSHEY_PLAIN,0.8, (255,255,255),1)
            if finalLabel != -1:
                cv2.putText(img,str(subject_dictionary[finalLabel]), (x,y), cv2.FONT_HERSHEY_PLAIN,0.8, (255,255,255),1)
            
            

        #Displaying image to screen with results        
        cv2.imshow('Recognition',img)

        #Press any key to continue to the next image. Exit out of shell to stop.
        cv2.waitKey(0)
        cv2.destroyAllWindows()
