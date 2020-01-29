


import argparse
from imutils import paths
from processors.AspectAwarePreporcessor import Resize
from processors.DataLoader import DataLoader
from processors.ImageToArray import ImageToArray
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from cnn.MiniVGGNet import MiniVGG
import numpy as np
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

#to run this script
# .\Flowers17VGG_createModel_unsingedData.py --output C:\gitProjects\perceptron\outputs\MiniVGGFlowers17_unsingedData.hdf5 --dataset C:\gitProjects\perceptron\datasets\flowers17

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-o", "--output", required=True, help="path to store the hdf5 model")
args = vars(ap.parse_args())


dataset = args["dataset"]
im_size = 64

#create list of path of images
pathImages = list(paths.list_images(dataset))

#load the images and apply the processors
#create the processors
p_size = Resize(im_size,im_size) #32 * 32 conserving AspectRadio
p_imArr = ImageToArray() #convert to desire shape

dataload = DataLoader(preprocessors=[p_size, p_imArr])

(images,labels) = dataload.load(pathImages)

#change the images type
images = images.astype("float") / 255.0

#split our data to have data to test the model
(trainX, testX, trainY, testY) = train_test_split(images,labels)

#generate the one hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#parameter to train the network
lr = .01
momentum = 0.9
epochs = 100
decay = lr/epochs
classes = np.unique(labels)
batchSize = 32
print(images.shape)
print(images[3])
#load the model 32,32,3,17
#images shape (1360, 64,64,3) (examples, height, widht, depth)
model = MiniVGG.build(im_size,im_size,images.shape[3],len(classes))

sgd = SGD(lr=lr, momentum=momentum, decay = decay, nesterov=True)

model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics=["accuracy"])

#generate the data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
H = model.fit_generator(aug.flow(trainX,trainY, batch_size=batchSize), validation_data=(testX,testY), steps_per_epoch=len(trainX)//batchSize, epochs= epochs)
#H = model.fit(testX, testY, batch_size=batchSize, validation_data=(testX, testY), epochs= epochs)

#save the model
model.save(args["output"])

#print prediction model
predictions = model.predict(testX,testY)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names==classes))
#plot the results
plt.style.use("ggplot")
plt.plot(np.arange(0,epochs), H.history["loss"], labels="loss")
plt.title("Loss / Accuracy")
figpath = os.path.split(args["output"])[0] + os.sep + "flowers17VGG_unsigned.jpg"
plt.savefig(figpath)
plt.show()





