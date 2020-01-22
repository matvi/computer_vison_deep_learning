

import argparse
from imutils import paths
from processors.DataLoader import DataLoader
from processors.SimplePreprocessor import Resize
from processors.ImageToArray import ImageToArray
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from cnn.MiniVGGNet import MiniVGG
import numpy as np
import matplotlib.pyplot as plt
#python
#python .\AnimalsVGG_SafeModel.py --dataset C:\gitProjects\perceptron\datasets\animals --model C:\gitProjects\perceptron\outputs\AnimalsMiniVGG_model.hdf5
learning_rate = .01
num_epochs = 100
im_size = 32
im_depth = 3 #3 for RGB image and 1 for gray scale
batchSize = 64 #the number of samples that SGD is going to take in each epoch to train the network

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="dir dataset")
ap.add_argument("-m", "--model", required=True, help="dir to save model")

args =vars(ap.parse_args())

# 1.- First we need to get the list of the images that we are going to to train the model.
datasetPath = args["dataset"]
#1.1- Get the path list of all the images using paths
imagesPath = list(paths.list_images(datasetPath))
#1.2 Get the numpy array and the corresponding label.
#In order to get that we will use our DataLoader.
#DataLoader needs to set up the preprocessors that will be appling to our images.
rs = Resize(im_size, im_size) #32*32
iarr = ImageToArray()
dataloader = DataLoader(preprocessors = [rs,iarr])
(images, labels) = dataloader.load(imagesPath)

#2.- prepare the data to train the model

#now that we have our images as numpy array and our labels it is time to split the data to train and test
(trainX, testX, trainY, testY) = train_test_split(images, labels)

# our images are in format [0-255] we need to change it to [0-1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#one hot encoding
# LabelBinarizer will transform our labels in format of one hot encoding so we can train our model
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#3 train the model

#3.1 get the optimizer
#decay: parameter to reduce the learning rate over time. A common setting is to set it as  learing_rate/num_epochs
#nesterov accelerated gradient
sgd = SGD(lr=learning_rate, decay=(learning_rate/num_epochs), nesterov=True, momentum=0.9)

#3.2 Load the model
#(weight, height, depth, classes)
classes = np.unique(labels).shape[0]
model = MiniVGG.build(im_size, im_size, im_depth, classes)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#3.3 Train the model
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size = batchSize, epochs = num_epochs)

#4.- Save the model
model.save(args["model"])

#plot the final result
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,num_epochs), H.history["loss"], label="tranning loss")
plt.title("Training Loss and Accuracy")
plt.show()

