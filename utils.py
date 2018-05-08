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