#In this example we are going to use Flower17 dataset
#This dataset has 17 classes of flowers. 
# This dataset only contains 80 images per class a total of 1360 images
# A general rule of thumb when applying deep learnin gto compter vision tasks is to have 1000-5000 examples per class
# With a small dataset our model is probably to overfit.
# To avoid overfitting due to a small data set we can use a regularization techinique calle data augmentation
# Data augmentation is a techinique used to generate new traning samples from th eoriginal ones by applying random jitters and perturbations like:
# Tranlations, rotations, changes in scale, shearing or Horizontal and vertical flips
# More advanced techniques can be applied like random perturbations of colos in a given color space and nonlinear geometric distortions.

import argparse
from processors.AspectAwarePreporcessor import Resize
from processors.DataLoader import DataLoader
from processors.ImageToArray import ImageToArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from cnn.MiniVGGNet import MiniVGG
import numpy as np
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import os

ap = argparse.ArgumentParser()
ap.add_argument("--images", "-i", required = True, help ="path to the dataset")
ap.add_argument("--output","-o", required = True, help= "path to store the model")

args = vars(ap.parse_args())

path = args["images"]
imagesPath = list(paths.list_images(path))
im_size = 32

processor_size = Resize(im_size,im_size)
processor_imArr = ImageToArray()
dataloader = DataLoader(preprocessors=[processor_size, processor_imArr])
(data, labels) = dataloader.load(imagesPath)

data = data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, labels, ran)

#one hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#model
classes = np.unique(labels)
epochs = 100
lr = .01
decadey = lr/epochs
momentum = 0.9
batchSize = 32

model = MiniVGG.build(im_size,im_size, data.shape[3], len(classes))

sgd = SGD(lr=lr, momentum= momentum, decay= decadey, nesterov=True)

model.compile(optimizer = sgd, loss= "categorical_crossentropy", metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batchSize, epochs=epochs)

model.save(args["output"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,epochs), H.history["loss"], label="tranning loss")
plt.title("Tranining Loss and Accuracy")
plt.show()
plotpath = os.path.split(args["output"])[0] + os.sep + "flowers17_plot.jpg"
print(plotpath)
plt.savefig(plotpath)
