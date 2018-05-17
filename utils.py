####################################################################################################
#                                            utlis.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: Julian Merkofer, Julian Moosmann, Selim Naji                                            #
#                                                                                                  #
# Purpose: Helpful functions...                                                                    #
#                                                                                                  #
#          crop needs the following xml file:                                                      #
#          https://github.com/shantnu/FaceDetect/blob/master/haarcascade_frontalface_default.xml   #
#                                                                                                  #
####################################################################################################


import cv2
import numpy as np


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


#****************************************************************#
#   resizes given data set to the given size and centers faces   #
#****************************************************************#
def crop(dataSet, size):

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        croppedSet = []

        for frame in dataSet:
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(size)
                # flags=cv2.CV_HAAR_SCALE_IMAGE
            )
            if(len(faces) != 0):
                for (x, y, w, h) in faces:
                    if y - 60 > 0 and y + h + 60 < frame.shape[0] and x - 20 > 0 \
                    and x + w + 20 < frame.shape[1]:
                        croppedSet.append(frame[(y - 60):(y + h + 60), (x - 20):(x + w + 20), :])
                    else:
                        croppedSet.append(frame)

            else:
                croppedSet.append(frame)

        return resize(croppedSet, size)
