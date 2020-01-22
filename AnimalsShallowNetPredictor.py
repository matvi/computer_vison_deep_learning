
#this class will read a model and will use it to predict the top 10 random values selected from animals dataset

import argparse
from keras.models import load_model
import os
from imutils import paths
import random
from processors.SimplePreprocessor import Resize
from processors.ImageToArray import ImageToArray
from processors.DataLoader import DataLoader
import matplotlib.pyplot as plt

#to run this script you need to run first the AnimalsShallowNet.py to create the AnimalsShallowNet_model.hdf5 that will be
#used to make the predictions and you need to have the dataset of animals (cat, dog, panda)

# python .\AnimalsShallowNetPredictor.py --dataset "C:\gitProjects\perceptron\datasets\animals" --model "C:\gitProjects\perceptron\outputs\AnimalsShalloNet_model.hdf5"

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help="Dataset dir")
ap.add_argument("-m", "--model", required=True, help="model dir")

args = vars(ap.parse_args())

#load list of direcction of images in subfolders
images = list(paths.list_images(args["dataset"]))

#shuffle images path
random.shuffle(images)
#we take only 10
images = images[0:10]

#we need to process the images the same way we proecced them for training

sp = Resize(32,32)
iarr = ImageToArray()

dataloader = DataLoader([sp,iarr])
(data, labels) = dataloader.load(images)
data = data.astype("float") / 255.0

#
animals = ["cat", "dog", "panda"]


#we load the model to make prediction on the data
model = load_model(args["model"])
predict = model.predict((data))

#print the results

for i in range(10):
    print("it is -> {}".format(labels[i]))
    print("prediction -> {}".format(animals[predict[i].argmax()]))
    plt.imshow(data[i])
    plt.title(animals[predict[i].argmax()])
    plt.show()



