####################################################################################################
#                                            mainrp.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: Julian Merkofer, Julian Moosmann, Selim Naji                                            #
#                                                                                                  #
# Purpose: How to Take a Good Selfie?  implementation to raspberry py                              #
#                                                                                                  #
# Version: 1.0                                                                                     #
#                                                                                                  #
####################################################################################################
# # import the necessary packages
# from picamera.array import PiRGBArray
# from picamera import PiCamera
# from keras.models import load_model      #
# import model.h5
# import time
# import cv2
#
# # initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# camera.resolution = (640, 480)
# camera.framerate = 15
# rawCapture = PiRGBArray(camera, size=(640, 480))
#
# # compute average popularityScore
# average = 0
# for i in range(dataSet.shape[0]):
#     average += float(dataSet[i, dict['popularityScore']])
# average /= dataSet.shape[0]
#
# # allow the camera to warmup
# time.sleep(0.2)
#
# # capture frames from the camera
# for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#   # grab the raw NumPy array representing the image, then initialize the timestamp
#   # and occupied/unoccupied text
#
#   selfiePoints = model.h5.predict(frame)
#   if selfiePoints > average:             #if selfie better than averagePopularityScore
#     cv2.imshow('goodSelfie', image)              #show picture on Monitor
#     cv2.imwrite('/goodSelfies/goodSelfie.png', image)       #safe picture in goodSelfies folder
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#   else:                                 #if worse than averagePopularityScore
#       cv2.imshow('badSelfie', image)    #show picture on Monitor
#       cv2.imwrite('/badSelfies/badSelfie.png', image) #safe picture in badSelfies folder
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#           break
#
#   # clear the stream in preparation for the next frame
#   rawCapture.truncate(0)


  ###############################################################
  #                    using webcam                             #
  ###############################################################
import numpy as np
import cv2
from keras.models import load_model      #
import model.h5
import time
import cv2

# open the camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    selfiePoints = model.h5.predict(frame)
    if selfiePoints > average:             #if selfie better than averagePopularityScore
        cv2.imshow('goodSelfie', image)              #show picture on Monitor
        cv2.imwrite('/goodSelfies/goodSelfie.png', image)       #safe picture in goodSelfies folder
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      else:                                 #if worse than averagePopularityScore
          cv2.imshow('badSelfie', image)    #show picture on Monitor
          cv2.imwrite('/badSelfies/badSelfie.png', image) #safe picture in badSelfies folder
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
