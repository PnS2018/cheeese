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

        bestImgs = [None] * 15
        bestAcc = np.ones((15,)) * 0.

        worstImgs = [None] * 5
        worstAcc = np.ones((5,)) * 1.

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

            # get 5 best and worst images
            indexBest = bestAcc.argmin()
            indexWorst = worstAcc.argmax()

            if acc[0][0] > bestAcc.min():
                bestAcc[indexBest] = acc[0][0]
                bestImgs[indexBest] = saveImg

            if acc[0][0] < worstAcc.max():
                worstAcc[indexWorst] = acc[0][0]
                worstImgs[indexWorst] = saveImg

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

        plt.figure()
        for i in xrange(5):
            plt.subplot(2, 5, i+1)
            if bestImgs is not None:
                plt.imshow(bestImgs[i])
            plt.title("certainty: %s" %(round(bestAcc[i])))

        for i in xrange(5):
            plt.subplot(2, 5, i+6)
            if worstImgs is not None:
                plt.imshow(worstImgs[i])
            plt.title("certainty: %s" %(round(1-worstAcc[i])))
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

        best = 0.
        worst = 1.

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

            if acc[0][0] > best:
                bestImg = saveImg
                best = acc[0][0]

            if acc[0][0] < worst:
                worstImg = saveImg
                worst = acc[0][0]

            cv2.imshow('yourself', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

            times.append(time.time() - tick)
            tick = time.time()

        cv2.destroyAllWindows()

        print "average time taken per frame is {} seconds".format(np.mean(times))

        print "\nbest accuracy:", best
        print "\nworst accuracy:", worst

        cv2.imshow('best image', bestImg)
        cv2.waitKey(0)

        cv2.imshow('worst image', worstImg)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        answer = raw_input("do you want to save current image (y/n)?")

        if answer == 'y':
            cv2.imwrite('selfies/bestSelfie.png', bestImg)
            cv2.imwrite('selfies/worstSelfie.png', worstImg)


#camera = PiCam(load_model('model.h5'), (32, 32))
#camera.show()
