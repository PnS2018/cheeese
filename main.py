####################################################################################################
#                                            Cheese.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: Julian Merkofer, Julian Moosmann, Selim Naji                                            #
#                                                                                                  #
# Purpose: How to Take a Good Selfie?                                                              #
#                                                                                                  #
# Version: 1.0                                                                                     #
#                                                                                                  #
####################################################################################################


import cv2
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, AvgPool2D
from keras.models import Model
from keras.utils import to_categorical
import time

import matplotlib.pyplot as plt

from utils import *



#**********************#
#   initializes time   #
#**********************#
tick = time.time()


#*******************************************#
#   defines column names for the data set   #
#*******************************************#
dict = {
    'imageName':0,
    'popularityScore':1,
    'partialFaces':2,
    'female':3,
    'baby':4,
    'child':5,
    'teenager':6,
    'youth':7,
    'middleAge':8,
    'senior':9,
    'white':10,
    'black':11,
    'asian':12,
    'ovalFace':13,
    'roundFace':14,
    'heartFace':15,
    'smiling':16,
    'mouthOpen':17,
    'frowning':18,
    'wearingGlasses':19,
    'wearingSunglasses':20,
    'wearingLipstick':21,
    'tongueOut':22,
    'duckFace':23,
    'blackHair':24,
    'blondHair':25,
    'brownHair':26,
    'redHair':27,
    'curlyHair':28,
    'straightHair':29,
    'braidHair':30,
    'showingCellphone':31,
    'usingEarphone':32,
    'usingMirror':33,
    'braces':34,
    'wearingHat':35,
    'harshLighting':36,
    'dimLighting':37
}


#********************#
#   loads data set   #
#********************#
dataSet = np.loadtxt('selfie_dataset.txt', dtype=object)

imgDataSet = np.load('selfie_dataset_64x64.npy')
print "time to load data set: ", time.time() - tick, "s"


#********************************************#
#   calculates average of popularity score   #
#********************************************#
average = 0
femaleLength = 0
maleLength = 0
femaleAverage = 0
maleAverage = 0

for i in range(dataSet.shape[0]):
    average += float(dataSet[i, dict['popularityScore']])
    if int(dataSet[i, dict['female']]) == 1:
        femaleAverage += float(dataSet[i, dict['popularityScore']])
        femaleLength += 1
    else:
        maleAverage += float(dataSet[i, dict['popularityScore']])
        maleLength += 1

average /= dataSet.shape[0]
femaleAverage /= femaleLength
maleAverage /= maleLength


#***********************************************************#
#   binaryfication of popularity score within two classes   #
#***********************************************************#
for i in range(dataSet.shape[0]):
    if int(dataSet[i, dict['female']]) == 1:
        if float(dataSet[i, dict['popularityScore']]) >= femaleAverage:
            dataSet[i, dict['popularityScore']] = 1
        else:
            dataSet[i, dict['popularityScore']] = 0

    else:
        if float(dataSet[i, dict['popularityScore']]) >= maleAverage:
            dataSet[i, dict['popularityScore']] = 1
        else:
            dataSet[i, dict['popularityScore']] = 0


#*******************************************#
#   splits data set in train and test set   #
#*******************************************#
trainX = dataSet[: int(imgDataSet.shape[0] * 0.9)]
testX = dataSet[int(imgDataSet.shape[0] * 0.9):]

imgTrainX = imgDataSet[: int(imgDataSet.shape[0] * 0.9)]
imgTestX = imgDataSet[int(imgDataSet.shape[0] * 0.9):]

trainY = to_categorical(trainX[:, dict['popularityScore']], num_classes = 2)

#***********************************#
#   initializes and defines model   #
#***********************************#
x = Input((imgTrainX.shape[1], imgTrainX.shape[2], imgTrainX.shape[3]))
y = Conv2D(filters=24, kernel_size=(7, 7), activation='relu')(x)
y = AvgPool2D(pool_size=(2, 2))(y)
y = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(y)
y = AvgPool2D(pool_size=(2, 2))(y)
y = Conv2D(filters=96, kernel_size=(3, 3), activation='relu')(y)
y = AvgPool2D(pool_size=(2, 2))(y)

y = Flatten()(y)

y = Dense(128, activation='relu')(y)
y = Dense(128, activation='relu')(y)
y = Dense(2, activation='softmax')(y)

model = Model(x, y)
model.summary()

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=imgTrainX, y=trainY, batch_size=256, epochs=1,
          validation_split=0.2)


#****************#
#   test model   #
#****************#
testXvis = imgTestX[:10]
groundTruths = testX[:10, dict['popularityScore']]

preds = (model.predict(testXvis) > 0.5)[:, 0].astype(np.int)

labels = ["bad", "good"]

plt.figure()
for i in xrange(2):
    for j in xrange(5):
        plt.subplot(2, 5, i * 5 + j + 1)
        plt.imshow(imgTestX[i*5+j], cmap = "gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[groundTruths[i * 5 + j]], labels[preds[i * 5 + j]]))
plt.show()
