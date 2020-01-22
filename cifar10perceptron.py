from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/acurracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR dataset...")

((trainX,trainY),(testX,testY)) = cifar10.load_data()

#convert it to floating point[0-1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#reshape the training data
#image shape = 32 * 32 * 3 = 3072
trainX = trainX.reshape((trainX.shape[0],3072)) #50000, 3072
testX = testX.reshape((testX.shape[0], 3072)) # 10000, 3072

#one hot encoding -> convert integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] Generating network architecture")
#define netwrok architecture
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training model")
#train the model
sgd = SGD(lr=0.01)
model.compile(sgd,"categorical_crossentropy",metrics=["accuracy"])
H = model.fit(trainX,trainY,epochs=100, batch_size=1, validation_data=(testX,testY))

print("[INFO] Evaluating network")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names = labels))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("epoch #")
plt.ylabel("Loss / Acurracy")
plt.legend()
plt.savefig(args["output"])