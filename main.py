####################################################################################################
#                                             main.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: Julian Merkofer, Julian Moosmann, Selim Naji                                            #
#                                                                                                  #
# Purpose: How to Take a Good Selfie?                                                              #
#          The deep learning model for our selfie classification is trained here.                  #
#                                                                                                  #
####################################################################################################


#import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import optimizers

#import cv2
#import numpy as np
import matplotlib.pyplot as plt
import time

from skimage import color
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

imgDataSet = np.load('selfie_dataset_32x32.npy')
print "time to load data set:", time.time() - tick, "s"
tick = time.time()

# uncomment code below to save data set in different sizes #
#**********************************************************#
# imgDataSet = resize(imgDataSet, (32, 32))
# print "resized..."
# np.save('selfie_dataset_32x32.npy', imgDataSet)
# print "saved..."

# uncomment code below to set color channel to 1 #
#************************************************#
imgDataSet = color.rgb2grey(imgDataSet)
imgDataSet = np.expand_dims(imgDataSet, axis=3)
print "time to set color channel to 1:", time.time() - tick, "s"
tick = time.time()

# imgDataSet = imgDataSet.astype('float32')/255.


#*******************************************#
#   splits data set in train and test set   #
#*******************************************#
trainX = dataSet[: int(imgDataSet.shape[0] * 0.9)]
testX = dataSet[int(imgDataSet.shape[0] * 0.9):]

imgTrainX = imgDataSet[: int(imgDataSet.shape[0] * 0.9)]
imgTestX = imgDataSet[int(imgDataSet.shape[0] * 0.9):]

mean = np.mean(imgTrainX, axis=0)

imgTrainX -= mean
imgTestX -= mean


#*********************************************#
#   calculates averages of popularity score   #
#*********************************************#
average = 0
femaleAverage = 0
maleAverage = 0

femaleLength = 0
maleLength = 0

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

    # if int(dataSet[i, dict['female']]) == 1:
    #     if float(dataSet[i, dict['popularityScore']]) >= femaleAverage:
    #         dataSet[i, dict['popularityScore']] = 1
    #     else:
    #         dataSet[i, dict['popularityScore']] = 0
    #
    # else:
    #     if float(dataSet[i, dict['popularityScore']]) >= maleAverage:
    #         dataSet[i, dict['popularityScore']] = 1
    #     else:
    #         dataSet[i, dict['popularityScore']] = 0

    # ignoring categorization
    if float(dataSet[i, dict['popularityScore']]) >= average:
        dataSet[i, dict['popularityScore']] = 1

    else:
        dataSet[i, dict['popularityScore']] = 0


#**************************************************************************#
#   converting the input class labels to categorical labels for training   #
#**************************************************************************#
# trainY = to_categorical(trainX[:, dict['popularityScore']], num_classes = 2)
# testY = to_categorical(testX[:, dict['popularityScore']], num_classes = 2)

# ignoring categorization
trainY = trainX[:, dict['popularityScore']]
testY = testX[:, dict['popularityScore']]


#*******************#
#   defines model   #
#*******************#
x = Input((imgTrainX.shape[1], imgTrainX.shape[2], imgDataSet.shape[3]))

# first approach #
#**#*************#
# y = Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(x)
# y = MaxPool2D(pool_size=(3, 3))(y)
# y = Dropout(rate=0.2)(y)
# y = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(y)
# y = MaxPool2D(pool_size=(3, 3))(y)
# y = Dropout(rate=0.2)(y)
# y = Conv2D(filters=196, kernel_size=(3, 3), activation='relu')(y)
# y = MaxPool2D(pool_size=(3, 3))(y)
# y = Dropout(rate=0.2)(y)
#
# y = Flatten()(y)
#
# y = Dense(128, activation='relu')(y)
# y = Dropout(rate=0.2)(y)
# y = Dense(128, activation='relu')(y)
# y = Dropout(rate=0.2)(y)
# y = Dense(1, activation='sigmoid')(y)


# complex approach #
#******************#
# y = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(x)
# y = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu')(y)
# y = MaxPool2D(pool_size=(2, 2), strides=(1, 1))(y)
# y = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),  activation='relu')(y)
# y = MaxPool2D(pool_size=(2, 2), strides=(1, 1))(y)
# y = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(y)
# y = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(y)
# y = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
#
# y = Flatten()(y)
#
# y = Dense(4096, activation='relu')(y)
# y = Dense(4096, activation='relu')(y)
# y = Dense(1, activation='sigmoid')(y)


# denseNet approach #
#*******************#
# y = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(x)
# y = Dense(320, activation='relu')(y)
# y = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(y)
# y = MaxPool2D(pool_size=(3, 3))(y)
# y = Dense(160, activation='relu')(y)
# y = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(y)
# y = MaxPool2D(pool_size=(3, 3))(y)
# y = Dense(80, activation='relu')(y)
# y = MaxPool2D(pool_size=(3, 3))(y)
# y = Flatten()(y)
# y = Dense(1, activation='sigmoid')(y)


# simple but most "successful" approach #
#***************************************#
y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
y = MaxPool2D(pool_size=(3, 3))(y)
y = Dropout(rate=0.2)(y)
y = Flatten()(y)
y = Dense(128, activation='relu')(y)
y = Dropout(rate=0.2)(y)
y = Dense(1, activation='sigmoid')(y)


#***********************#
#   initializes model   #
#***********************#
model = Model(x, y)
model.summary()

sgd = optimizers.SGD(lr=0.01)
adam = optimizers.Adam(lr=0.0001)

checkpoint = ModelCheckpoint(filepath='model.h5', save_best_only=True)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

batchSize = 32      # assures that same batch size is used while evaluating
model.fit(x=imgTrainX, y=trainY, batch_size=batchSize, epochs=50, validation_split=0.2,
          callbacks=[checkpoint])
print "time spent on training:", time.time() - tick, "s"
tick = time.time()


#*****************#
#   tests model   #
#*****************#
accuracy = model.evaluate(imgTestX, testY, batch_size=batchSize)[1]
print "test accuracy:", accuracy


#*****************************************#
#   plots images with their predictions   #
#*****************************************#
testXvis = imgTestX[:15]
groundTruths = testX[:15, dict['popularityScore']]

preds = (model.predict(testXvis) > 0.5)[:, 0].astype(np.int)

labels = ["bad", "good"]

# removes dimension for plotting (use when color channel 1)
imgTestX = np.squeeze(imgTestX, axis=3)

# uncomment code below to plot images...
# plt.figure()
# for i in xrange(3):
#     for j in xrange(5):
#         plt.subplot(3, 5, 5*i + j + 1)
#         plt.imshow(imgTestX[5*i + j], cmap="gray")
#         plt.title("Ground Truth: %s, \n Prediction %s" %
#                   (labels[groundTruths[5*i + j]], labels[preds[5*i + j]]))
# plt.show()
