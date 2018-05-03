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


#import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Model
from keras.utils import to_categorical
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
#imgDataSet = resize(imgDataSet, (32, 32))
#print "resized..."
#np.save('selfie_dataset_32x32.npy', imgDataSet)
#print "saved..."

# uncomment code below to set color channel to 1 #
#************************************************#
#imgDataSet = color.rgb2grey(imgDataSet)
#imgDataSet = np.expand_dims(imgDataSet, axis=3)
#print "time to set color channel to 1:", time.time() - tick, "s"
#tick = time.time()

imgDataSet = imgDataSet.astype('float32')/255.


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
    
    #if int(dataSet[i, dict['female']]) == 1:
    #if float(dataSet[i, dict['popularityScore']]) >= femaleAverage:
    #dataSet[i, dict['popularityScore']] = 1
    else:
        dataSet[i, dict['popularityScore']] = 0
    
    else:
        if float(dataSet[i, dict['popularityScore']]) >= maleAverage:
            dataSet[i, dict['popularityScore']] = 1
        else:
            dataSet[i, dict['popularityScore']] = 0

# ignoring categorization
#if float(dataSet[i, dict['popularityScore']]) >= average:
#dataSet[i, dict['popularityScore']] = 1

#else:
#dataSet[i, dict['popularityScore']] = 0




#**************************************************************************#
#   converting the input class labels to categorical labels for training   #
#**************************************************************************#
trainY = to_categorical(trainX[:, dict['popularityScore']], num_classes = 2)

# ignoring categorization
#trainY = trainX[:, dict['popularityScore']]


#*******************#
#   defines model   #
#*******************#
x = Input((imgTrainX.shape[1], imgTrainX.shape[2], imgDataSet.shape[3]))

# first naive try #
#*****************#
# ~ 50% accuracy (no matter what)

#y = Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(x)
#y = MaxPool2D(pool_size=(3, 3))(y)
#y = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(y)
#y = MaxPool2D(pool_size=(3, 3))(y)
#y = Conv2D(filters=192, kernel_size=(3, 3), activation='relu')(y)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

#y = Dense(256, activation='relu')(y)
#y = Dense(256, activation='relu')(y)
#y = Dense(2, activation='softmax')(y)


# simple approach #
#*****************#
# ~ 66% accuracy (32x32, batch=128, epochs=10, t > 300s)
# ~ 68% accuracy (64x64, batch=128, epochs=10, t > 2200s)

#y = Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(x)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

#y = Dense(128, activation='relu')(y)
#y = Dense(2, activation='softmax')(y)


# "successful" approach #
#***********************#
# ??? overfitting... (what can we do? bigger data set?)

# ~ 66% accuracy (32x32, batch=128, epochs=10, t > 75s)
# ~ 74% accuracy, ~ 69% val_acc (32x32, batch=16, epochs=20, t > 300s)
# ~ 99% accuracy, ~ 64% val_acc (32x32, batch=16, epochs=60, t > 800s)

# ~ 67% accuracy (64x64, batch=128, epochs=10, t > 300s)
# ~ 72% accuracy, ~ 69% val_acc (128x128, batch=16, epochs=10, t > 300s)
# ~ 88% accuracy, ~ 65% val_acc (64x64, batch=16, epochs=20, t > 750s)

# ~ 80% accuracy, ~ 68% val_acc (128x128, batch=16, epochs=10, t > 2000s)
# ~ 99% accuracy, ~ 65% val_acc (128x128, batch=16, epochs=20, t > 3600s)

#y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

#y = Dense(128, activation='relu')(y)
#y = Dense(2, activation='softmax')(y)


# "successful" approach, trying not to overfit #
#**********************************************#
# ??? converges to values below...

# ~ 70% accuracy, ~ 69% val_acc (32x32, batch=16, epochs=20, t > 200s)
# ~ 71% accuracy, ~ 68% val_acc (64x64, batch=16, epochs=20, t > 600s)
# ~ 76% accuracy, ~ 65% val_acc (128x128, batch=16, epochs=20, t > 3000s)

## ~ 72% accuracy, ~ 68% val_acc (32x32, batch=16, epochs=20, t > 1000s)

##y = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
##y = MaxPool2D(pool_size=(3, 3))(y)
#y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(y)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

##y = Dense(64, activation='relu')(y)
#y = Dense(2, activation='softmax')(y)


# new approach (loading image data set as float32 and dividing it by 255) #
#*************************************************************************#
# ~ 68% accuracy, ~ 68% val_acc (32x32 COLOR, batch=128, epochs=10, t > 750s)
# ~ 68% accuracy, ~ 67% val_acc (64x64 COLOR, batch=128, epochs=10, t > 8000s)

# ~ 86% accuracy, ~ 64% val_acc (32x32 COLOR, batch=16, epochs=20, t > 1500s)
# -> after 16 epochs (acc at 80% and val_acc 69%) the val_acc starts to drop

#y = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

#y = Dense(128, activation='relu')(y)
#y = Dense(128, activation='relu')(y)
#y = Dense(2, activation='softmax')(y)


# desperate approaches #
#**********************#
# ~ 70% accuracy, ~ 67% val_acc (64x64 COLOR and fl32/255, batch=128, epochs=50, t > 12000s)
# ~ 70% accuracy, ~ 67% val_acc (32x32 COLOR and fl32/255, batch=16, epochs=20, t > 2400s)
#y = Conv2D(filters=16, kernel_size=(7, 7), activation='relu')(x)
#y = MaxPool2D(pool_size=(3, 3))(y)
#y = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(y)
#y = MaxPool2D(pool_size=(3, 3))(y)
#y = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(y)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

#y = Dense(64, activation='relu')(y)
#y = Dense(64, activation='relu')(y)
#y = Dense(2, activation='softmax')(y)


# testing dropout approach #
#**************************#

#y = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
#y = MaxPool2D(pool_size=(3, 3))(y)

#y = Flatten()(y)

#y = Dense(64, activation='relu')(y)
#y = Dropout(rate=0.2)(y)
#y = Dense(1, activation='sigmoid')(y)


# testing denseNet #
#******************#

y = Conv2D(filters=8, kernel_size=(7, 7), activation='relu')(x)
y = Dense(160, activation='relu')(y)
y = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(y)
y = MaxPool2D(pool_size=(2, 2))(y)
y = Dense(80, activation='relu')(y)#
y = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(y)
y = MaxPool2D(pool_size=(2, 2))(y)
y = Dense(40, activation='relu')(y)
y = MaxPool2D(pool_size=(2, 2))(y)
y = Flatten()(y)
y = Dense(2, activation='softmax')(y)


#***********************#
#   initializes model   #
#***********************#
model = Model(x, y)
model.summary()

sgd = optimizers.SGD(lr=0.01)
adam = optimizers.Adam(lr=0.0001)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=imgTrainX, y=trainY, batch_size=128, epochs=30, validation_split=0.2)
print "time spent on training:", time.time() - tick, "s"
tick = time.time()

model.save('model.h5')


#*****************#
#   tests model   #
#*****************#
correct = 0
testXvis = imgTestX
for i in range(testXvis.shape[0]):
    # ??? why is this so slow?
    correct += (model.predict(testXvis, batch_size=128) > 0.5)[:, 0].astype(np.int)[i]

print "accuracy of model with test set:", float(correct) / testXvis.shape[0],\
    "(", time.time() - tick, "s )"


#*****************************************#
#   plots images with their predictions   #
#*****************************************#
testXvis = imgTestX[:15]
groundTruths = testX[:15, dict['popularityScore']]

preds = (model.predict(testXvis) > 0.5)[:, 0].astype(np.int)

labels = ["bad", "good"]

# removes dimension for plotting
#imgTestX = np.squeeze(imgTestX, axis=3)

plt.figure()
for i in xrange(3):
    for j in xrange(5):
        plt.subplot(3, 5, 5*i + j + 1)
        plt.imshow(imgTestX[5*i + j], cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[groundTruths[5*i + j]], labels[preds[5*i + j]]))
plt.show()

