#This call will take an already trained model and will use it to predict 
#It will take the number given as a parameter and look for aleatory images in the dataset
# To run this class execute:
#  python AnimalsVGG_predictor.py --model C:\gitProjects\perceptron\outputs\AnimalsMiniVGG_model.hdf5 --number 10 --dataset C:\gitProjects\perceptron\datasets\animals

import argparse
from imutils import paths
import random
from processors.DataLoader import DataLoader
from processors.ImageToArray import ImageToArray
from processors.SimplePreprocessor import Resize
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required= True, help="dir model hdf5 format")
ap.add_argument("-n","--number", required= True, help="number of images to predict")
ap.add_argument("-d","--dataset", required = True, help= "path to the images")

#vars returns a dictionary of the object given so we can recover the values given as parameters
args = vars(ap.parse_args())
numImages = int(args["number"])
dataset = args["dataset"]
pathModel = args["model"]

#We need to load the images
imagesPaths = list(paths.list_images(dataset))
#we randome the list and take only the number that we want to predict
random.shuffle(imagesPaths)
imagesPaths = imagesPaths[0:numImages]
#now we need to preccess the images like we did when training
processor_rs = Resize(32,32)
processor_iarr = ImageToArray()
dataloader = DataLoader([processor_rs, processor_iarr])
(images, labels) = dataloader.load(imagesPaths)

images = images.astype("float") / 255.0

#we load the model so we can make predictions
model = load_model(pathModel)
predictions = model.predict(images)

# in the case of using AnimalsDataset the result is ["cat", "dog", "panda"]
uniqueLabels = np.unique(labels)

for (i,image) in enumerate(images):
    predicted = uniqueLabels[predictions[i].argmax()]
    actualObject = labels[i]
    title = "Predicted -> " + predicted + " ; Actual -> " + actualObject
    print("predicted -> {}".format(uniqueLabels[predictions[i].argmax()]))
    print("it is -> {}".format(labels[i]))
    plt.imshow(image)
    plt.title(title)
    plt.show()


