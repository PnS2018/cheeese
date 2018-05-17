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
import time
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

        times = []
        tick = time.time()

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

            times.append(time.time() - tick)
            tick = time.time()

        cap.release()
        cv2.destroyAllWindows()

        print "average time taken per frame is {} seconds".format(np.mean(times))

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


camera = WebCam(load_model('model.h5'), (32, 32))
camera.show()



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
        camera.resolution = (320, 240)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(320, 240))

        best = 0.
        worst = 1.

        times = []
        model_times = []
        start_time = time.time()

        cam_start_time = time.time()

        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            print("Camera took {} seconds".format(time.time() - cam_start_time))

            display = frame.array

            current_start = time.time()
            img = cv2.resize(display, dsize=self.size)
            print("Resize took {}".format(time.time() - current_start))

            current_start = time.time()
            img = color.rgb2grey(img)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)
            print("Kill color took {}".format(time.time() - current_start))

            model_start_time = time.time()
            acc = self.model.predict(img, batch_size=1)
            model_end_time = time.time()
            model_times.append(model_end_time - model_start_time)

            print "accuracy:", acc[0][0]

            if acc[0][0] > best:
                bestImg = display
                best = acc[0][0]

            if acc[0][0] < worst:
                worstImg = display
                worst = acc[0][0]

            cv2.imshow('yourself', img[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

            end_time = time.time()

            print("Current frame took {} seconds.".format(end_time - start_time))
            times.append(end_time - start_time)

            start_time = time.time()
            cam_start_time = time.time()

        cv2.destroyAllWindows()

        print "\nbest accuracy:", best
        print "\nworst accuracy:", worst

        print("Average time taken per frame is {} seconds and standard deviation is {} over {} images.".format(
            np.mean(times),
            np.std(times), len(times)))

        print("Average model time taken per frame is {} seconds and standard deviation is {} over {} images.".format(
            np.mean(model_times),
            np.std(model_times), len(model_times)))

        cv2.imshow('best image', bestImg)
        cv2.waitKey(0)

        cv2.imshow('worst image', worstImg)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        answer = raw_input("do you want to save current image (y/n)?")

        if answer == 'y':
            cv2.imwrite('selfies/selfie.png', bestImg)


#camera = PiCam(load_model('model.h5'), (32, 32))
#camera.show()
