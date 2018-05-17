####################################################################################################
#                                            utlis.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: Julian Merkofer, Julian Moosmann, Selim Naji                                            #
#                                                                                                  #
# Purpose: Helpful functions...                                                                    #
#                                                                                                  #
####################################################################################################


import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#*********************************************#
#   saves given data set as "name.npy" file   #
#*********************************************#
def jpgToNpy(dataSet, dict, name):
    images = [cv2.imread("./Selfie-dataset/images/" + str(fname) + ".jpg")
              for fname in dataSet[:, dict['imageName']]]

    np.save(name + '.npy', images)


#**********************************************#
#   resizes given data set to the given size   #
#**********************************************#
def resize(dataSet, size):
    return [cv2.resize(image, dsize=size) for image in dataSet]

def crop(dataSet, size):

        croppedSet = []
        for frame in dataSet:
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(size)
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            if(faces):
                for (x, y, w, h) in faces[0]:
                    cropImg = frame[(y - 60):(y + h + 60), (x - 20):(x + w + 20)]

                cropImg = cv2.resize(cropImg, dsize=size)
                croppedSet.append(cropImg)
            else:
                print("something went wrong!")

        return dataSet
