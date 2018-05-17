####################################################################################################
#                                              cam.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: Julian Merkofer, Julian Moosmann, Selim Naji                                            #
#                                                                                                  #
# Purpose: The interfaces of the web cam and the raspberry pi camera are implemented here          #
#          (not sure where to put them... yet)                                                     #
#                                                                                                  #
####################################################################################################


#from picamera.array import PiRGBArray
#from picamera import PiCamera

from keras.models import load_model

from skimage import color

import numpy as np
import cv2



#**************************************************************************************************#
#                                          Class WebCam                                            #
#**************************************************************************************************#
#                                                                                                  #
# Implements the interface of the web cam.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class WebCam():

    #*********************************************************#
    #   constructs the camera with all necessary properties   #
    #*********************************************************#
    def __init__(self, model, size):
        self.model = model
        self.size = size


    #********************************************#
    #   activates camera and shows predictions   #
    #********************************************#
    def show(self):
        cap = cv2.VideoCapture(0)
        best = 0.
        worst = 1.

        while (True):
            ret, frame = cap.read()
            img = cv2.resize(frame, dsize=self.size)

            img = color.rgb2grey(img)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            acc = self.model.predict(img, batch_size=1)
            print "accuracy:", acc[0][0]

            if acc[0][0] > best:
                bestImg = frame
                best = acc[0][0]

            if acc[0][0] < worst:
                worstImg = frame
                worst = acc[0][0]

            cv2.imshow('yourself', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

        print "\nbest accuracy:", best
        print "\nworst accuracy:", worst

        cv2.imshow('best image', bestImg)
        cv2.waitKey(0)

        cv2.imshow('worst image', worstImg)
        cv2.waitKey(0)

        cv2.destroyAllWindows()



        answer = raw_input("do you want to save current image (y/n)? ")

        if answer == 'y':
            cv2.imwrite('selfies/selfie.png', bestImg)


#camera = WebCam(load_model('model.h5'), (32, 32))
#camera.show()



#**************************************************************************************************#
#                                           Class PiCam                                            #
#**************************************************************************************************#
#                                                                                                  #
# Implements the interface of the raspberry pi cam. (NOT TESTED)                                   #
#                                                                                                  #
#**************************************************************************************************#
class PiCam():

    #*********************************************************#
    #   constructs the camera with all necessary properties   #
    #*********************************************************#
    def __init__(self, model, size):
        self.model = model
        self.size = size


    #********************************************#
    #   activates camera and shows predictions   #
    #********************************************#
    def show(self):
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 15
        rawCapture = PiRGBArray(camera, size=(640, 480))

        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            img = cv2.resize(frame, dsize=self.size)

            img = color.rgb2grey(img)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            acc = self.model.predict(img, batch_size=1)
            print "accuracy:", acc[0][0]

            if acc[0][0] > best:
                bestImg = frame
                best = acc[0][0]

            cv2.imshow('yourself', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

        cv2.destroyAllWindows()

        print "best accuracy:", best

        cv2.imshow('best image', bestImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        answer = raw_input("do you want to save current image (y/n)?")

        if answer == 'y':
            cv2.imwrite('selfies/selfie.png', bestImg)


camera = PiCam(load_model('model.h5'), (32, 32))
camera.show()