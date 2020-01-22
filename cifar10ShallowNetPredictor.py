

from keras.datasets import cifar10
import argparse
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# This class will use a trained model to make predictions -> to generate the model run first cifar10ShallowNetSafeModel.py
# To run this class execute: 
# python .\cifar10ShallowNetPredictor.py --model C:\gitProjects\perceptron\outputs\cifar10ShallowNet_model.hdf5

#we need the direction to the model.
ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",help="direction of the model to load")
arg = vars(ap.parse_args())

model = load_model(arg["model"])

# load the data that we´ll use to test
((trainX, trainY),(testX,testY)) = cifar10.load_data()

# convert the input data to be equal as the one we used to train
trainX = trainX.astype("float") / 255
testX = testX.astype("float") / 255 

#make the predictions
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
predict = model.predict(testX, batch_size=32)

for i in range (10) :
    print('position -> {}'.format( testY[i][0]))
    print('It is  ->' +labels[ testY[i][0] ])
    print('predicted ->' +labels[ predict[i].argmax() ])
    image = testX[i]
    plt.imshow(image)
    plt.title(labels[predict[i].argmax()])
    plt.show()





