# -*- coding: utf-8 -*-
#Names of students:
#Aya Khaled Mohamed Abdel-Raheem     20198016
#Yousef Ahmed Mohamed Ibraheem       20198103
#Amira Hamdy Sayed                   20198013
# For Reading Data
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
# Neural Network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Flatten, Dense
# SVM
from sklearn import svm
from sklearn import metrics

# Read Images
parentFolder = r"E:\\faculty toturials\\Year 3\sem 1\\Machine Learning\\assigments\\ass2\\Assignment 2 - Dataset (1)\\Assignment 2 - Dataset\\Dataset"
imagesData = np.empty(shape=(0, 100, 100))
labels = []

for i in os.listdir(parentFolder):
    classifingFolder = parentFolder + "/" + i
    for j in os.listdir(classifingFolder):
        imagePath = classifingFolder + '/' + j
        grayImage = np.array(Image.open(imagePath).convert('L'))
        colorThreshold = 128
        blackWhiteImage = (grayImage > colorThreshold) * 255
        imagesData = np.append(imagesData, [blackWhiteImage], axis=0)
        if i == 'Positive':
            labels.append(1)
        elif i == 'Negative':
            labels.append(0)


# Normalize ImagesData
imagesData /= 255

# Split Data to train and test
labels = np.array(labels)
xTrain, xTest, yTrain, yTest = train_test_split(
    imagesData, labels, test_size=0.20, random_state=42)

imagesDataShape = imagesData.shape
print("NNNNNNNNNNNNNNNNNN         ",xTrain.shape, xTest.shape)
trainsCount, testsCount = len(xTrain), len(xTest)

# Define the architecture of the neural network
model = Sequential()
model.add(Conv1D(16, 2, input_shape=imagesDataShape[1:]))

model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(16, 2))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(32, 2))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(32, 2))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

# Compile the keras model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit our data against the model
BATCH_SIZE = 32
print(xTrain.shape)
history = model.fit(
    xTrain,
    yTrain,
    epochs=40,
    steps_per_epoch=trainsCount // BATCH_SIZE,
    validation_data=(xTest, yTest),
    validation_steps=testsCount // BATCH_SIZE,
    verbose=0)

_,accuracy=model.evaluate(xTest,yTest)
print('Accuracy:',accuracy%100)

# Flatten our data for SVM model
xTrain = np.reshape(xTrain, (trainsCount, 100*100))
xTest = np.reshape(xTest, (testsCount, 100*100))

# Setup and fit SVM model to our data
svmModel = svm.SVC(kernel='sigmoid', gamma='auto')
svmModel.fit(xTrain, yTrain)
yPredict = svmModel.predict(xTest)
print("Support Vector Machine Model Accuracy: ",
      metrics.accuracy_score(yTest, yPredict))
