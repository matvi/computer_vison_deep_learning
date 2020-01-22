import argparse
from imutils import paths
from processors.DataLoader import DataLoader
from processors.ImageToArray import ImageToArray
from processors.SimplePreprocessor import Resize
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from cnn.MiniVGGNet import MiniVGG
import matplotlib.pyplot as plt
import numpy as np


#this script will train a MiniVGG network on smiles. It will serialize the model in a hdf5 file.
# to run this script run:
#  python .\SmileVGG_modelCreator.py --dataset C:\gitProjects\perceptron\datasets\SMILEsmileD --save C:\gitProjects\perceptron\outputs\MiniVGGSmiles_model.hdf5


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help = "paht to the dataset")
ap.add_argument("-s","--save", required=True, help = "path where the weights are going to be stored")
args = vars(ap.parse_args())

datasetPath = args["dataset"]

#the dataset it is formed by conformed by 13,165 graysacale images with each image having a size of 64*64

#this will get all the images path. example: /home/david/datasets/smiles/positives/1.jpg
imagesPaths = list(paths.list_images(datasetPath))

#I need to process the images
processorImArr = ImageToArray()
processorSize = Resize(32,32)
dataloader = DataLoader(preprocessors=[processorSize,processorImArr])
(data, labels) = dataloader.load(imagesPaths)

uniqLabels = np.unique(labels)
print(uniqLabels)
#convert the data type of the images
#data = data.astype("float") / 255.0

#(trainX, testX, trainY, testY) = train_test_split(data, labels)
#>>> testY
#array(['negatives', 'negatives', 'negatives', ..., 'positives',
#       'negatives', 'positives'], dtype='<U9')
#
#


#convert the labels to one hot encoding with LabelBinarizer
#lb = LabelBinarizer()
#trainY = lb.fit_transform(trainY)
#testY = lb.transform(testY)
#>>> testY
#LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
#>>> trainY
#array([[0],
#       [1],
#       [1],
#       ...,
#       [0],
#       [0],
#       [0]])

#convert the labels to one hot encoding with LabelEncoder
lb = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(lb.transform(labels), 2)
#>>> labels_ohe
#array([[1., 0.],
#       [1., 0.],
#       [1., 0.],
#       ...,
#       [0., 1.],
#       [0., 1.],
#       [0., 1.]], dtype=float32)

(trainX, testX, trainY, testY) = train_test_split(data, labels)

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#Now that we have our data ready we need to train the network, but before we need to call the model and set the
# parameters
numEpochs = 100
learningRate = .01
decadey = learningRate/numEpochs
optimizer = "adam"
momentum = 0.9
batchSize = 64

sgd = SGD(lr = learningRate, momentum= 0.9, decay = decadey, nesterov=True )

#we load the VGG model
model = MiniVGG.build(32,32,3,2)
#model.compile(loss="binary_crossentropy", optimizer = sgd, metrics = ["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])

H = model.fit(trainX, trainY, epochs=numEpochs, batch_size=batchSize, validation_data=(testX, testY))

model.save(args["model"])

plt.style.use("ggplot")
plt.plot(np.range[0:numEpochs], H.history["loss"], label = "training loss")
plt.title("Tranning VGG on Smiles")
plt.show()