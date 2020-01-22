#This script will test the flowers17 VGG model that was created using the file flowers17VGG_safemodel.py
#In order to see how efficient it is will be generate new images using dataAugmentation to test our model.

#To run the file
#python .\Flower17VGG_predictor.py --dataset C:\gitProjects\perceptron\datasets\flowers17 --model C:\gitProjects\perceptron\outputs\MiniVGGFlowers17.hdf5
import argparse
from keras.models import load_model
from imutils import paths
import random
from processors.AspectAwarePreporcessor import Resize
from processors.DataLoader import DataLoader
from processors.ImageToArray import ImageToArray
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help= "path to the dataset")
ap.add_argument("-m", "--model", required=True, help="path to the VGG model serialized as hdf5 ")

args = vars(ap.parse_args())

dataset = args["dataset"]
#create a list with all the images and select 10 random of those
pathImages = list(paths.list_images(dataset))
random.shuffle(pathImages)


#we need to process the images so we can feed the model
#first we create the processors
p_size = Resize(32,32)
p_iArr = ImageToArray()
dataloader = DataLoader(preprocessors=[p_size, p_iArr])
#second we apply the processors to the images, the result are the images Resized and as numpy arrays with their corresponding labels.
(images, labels) = dataloader.load(pathImages)
#convert the images from int to float [0,1]
images = images.astype("float") / 255.0

#get only 10 images
images = images[0:10]
print(images.shape)

#load the model
model = load_model(args["model"])

predictor = model.predict(images)

#classes will contain the classes ordered like follow
#['bluebell' 'buttercup' 'coltsfoot' 'cowslip' 'crocus' 'daffodil' 'daisy'
# 'dandelion' 'fritillary' 'iris' 'lilyvalley' 'pansy' 'snowdrop'
# 'sunflower' 'tigerlily' 'tulip' 'windflower']
classes = np.unique(labels)

for (i,image) in enumerate(images):
    predicted = classes[predictor[i].argmax()]
    actual = labels[i]
    plt.imshow(image)
    plt.title("predicted -> " + predicted + " ... actual -> " + actual)
    plt.show()
