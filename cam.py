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

import matplotlib.pylab as plt
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
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)

        bestImgs = [None] * 3
        bestAcc = np.ones((3,)) * 0.

        worstImgs = [None] * 3
        worstAcc = np.ones((3,)) * 1.

        i = 0

        times = []
        tick = time.time()

        while (True):
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)

            display = frame.copy()
            saveImg = frame.copy()

            # detect faces
            if i % 10 == 0:
                face = faceCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(self.size)
                    # flags=cv2.CV_HAAR_SCALE_IMAGE
                )
            i += 1

            for (x, y, w, h) in face:
                if y - 60 > 0 and y + h + 60 < frame.shape[0] and x - 20 > 0 \
                and x + w + 20 < frame.shape[1]:
                    frame = frame[(y - 60):(y + h + 60), (x - 20):(x + w + 20), :]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img = cv2.resize(frame, dsize=self.size)

            img = color.rgb2grey(img)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            acc = self.model.predict(img, batch_size=1)
            print "accuracy:", acc[0][0]

            # get 3 best and worst images
            indexMin = bestAcc.argmin()
            indexMax = worstAcc.argmax()

            if acc[0][0] > bestAcc.min():
                bestAcc[indexMin] = acc[0][0]
                bestImgs[indexMin] = saveImg

            if acc[0][0] < worstAcc.max():
                worstAcc[indexMax] = acc[0][0]
                worstImgs[indexMax] = saveImg

            # display stuff
            cv2.imshow('yourself', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            times.append(time.time() - tick)
            tick = time.time()

        cap.release()
        cv2.destroyAllWindows()

        print "average time taken per frame is {} seconds".format(np.mean(times))

        print "\nbest accuracy:", bestAcc
        print "\nworst accuracy:", worstAcc

        plt.figure("test")
        for i in xrange(3):
            plt.subplot(2, 3, i+1)
            plt.subplots_adjust(left=0.03, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
            if bestImgs is not None:
                plt.imshow(cv2.cvtColor(bestImgs[i], cv2.COLOR_BGR2RGB))
            plt.title("GOOD   certainty: %s" %(round(bestAcc[i], 2)))

        for i in xrange(3):
            plt.subplot(2, 3, i+4)
            plt.subplots_adjust(left=0.03, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
            if worstImgs is not None:
                plt.imshow(cv2.cvtColor(worstImgs[i], cv2.COLOR_BGR2RGB))
            plt.title("BAD   certainty: %s" %(round(1 - worstAcc[i], 2)))
        plt.show()

        answer = raw_input("do you want to save current images (y/n)? ")

        if answer == 'y':
            cv2.imwrite('selfies/bestSelfie.png', bestImgs[0])
            cv2.imwrite('selfies/worstSelfie.png', worstImgs[0])


camera = WebCam(load_model('simpleModel.h5'), (32, 32))
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

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        bestImgs = [None] * 3
        bestAcc = np.ones((3,)) * 0.

        worstImgs = [None] * 3
        worstAcc = np.ones((3,)) * 1.

        i = 0

        times = []
        tick = time.time()

        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            frame = cv2.flip(frame, 1)
            frame = frame.array

            display = frame.copy()
            saveImg = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i % 10 == 0:
                face = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(int(32), int(32))
                    # flags=cv2.CV_HAAR_SCALE_IMAGE
                )
            i += 1

            for (x, y, w, h) in face:
                if y - 60 > 0 and y + h + 60 < frame.shape[0] and x - 20 > 0 \
                        and x + w + 20 < frame.shape[1]:
                    frame = frame[(y - 60):(y + h + 60), (x - 20):(x + w + 20), :]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img = cv2.resize(frame, dsize=self.size)

            img = color.rgb2grey(img)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            acc = self.model.predict(img, batch_size=1)

            print "accuracy:", acc[0][0]

            # get 3 best and worst images
            indexMin = bestAcc.argmin()
            indexMax = worstAcc.argmax()

            if acc[0][0] > bestAcc.min():
                bestAcc[indexMin] = acc[0][0]
                bestImgs[indexMin] = saveImg

            if acc[0][0] < worstAcc.max():
                worstAcc[indexMax] = acc[0][0]
                worstImgs[indexMax] = saveImg

            cv2.imshow('yourself', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

            times.append(time.time() - tick)
            tick = time.time()

        cv2.destroyAllWindows()

        print "average time taken per frame is {} seconds".format(np.mean(times))

        print "\nbest accuracy:", bestAcc
        print "\nworst accuracy:", worstAcc

        plt.figure("test")
        for i in xrange(3):
            plt.subplot(2, 3, i + 1)
            plt.subplots_adjust(left=0.03, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
            if bestImgs is not None:
                plt.imshow(cv2.cvtColor(bestImgs[i], cv2.COLOR_BGR2RGB))
            plt.title("GOOD   certainty: %s" % (round(bestAcc[i], 2)))

        for i in xrange(3):
            plt.subplot(2, 3, i + 4)
            plt.subplots_adjust(left=0.03, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
            if worstImgs is not None:
                plt.imshow(cv2.cvtColor(worstImgs[i], cv2.COLOR_BGR2RGB))
            plt.title("BAD   certainty: %s" % (round(1 - worstAcc[i], 2)))
        plt.show()

        answer = raw_input("do you want to save current image (y/n)?")

        if answer == 'y':
            cv2.imwrite('selfies/bestSelfie.png', bestImgs[0])
            cv2.imwrite('selfies/worstSelfie.png', worstImgs[0])


#camera = PiCam(load_model('model.h5'), (32, 32))
#camera.show()
