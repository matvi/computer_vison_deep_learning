

import argparse
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from cnn.ShallowNet import Shallownet
import matplotlib.pyplot as plt
from processors.SimplePreprocessor import Resize
from processors.DataLoader import DataLoader
import numpy as np
from processors.ImageToArray import ImageToArray
from sklearn.preprocessing import LabelBinarizer

#To run this class execute this:
# python.exe .\AnimalsShallowNet.py --dataset C:\gitProjects\perceptron\datasets\animals --model_dir C:\gitProjects\perceptron\outputs\AnimalsShalloNet_model.hdf5

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required = True, help="Dataset path")
ap.add_argument("-m", "--model_dir", required = True, help="dir to save model")

args = vars(ap.parse_args())



path = args["dataset"]
imagePaths = list(paths.list_images(path))

sp = Resize(32,32)
iap = ImageToArray()

sdl = DataLoader(preprocessors=[sp,iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size = 0.25, random_state=42)

#convert labels to one hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

opt = SGD(lr=.01)
model = Shallownet.build(32,32,3,3) #width, height, depth, classes

model.compile(loss="categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size=32, epochs= 100, verbose=1)
model.save(args["model_dir"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="tranning loss")
plt.title("Training Loss and Accuracy")
plt.show()








