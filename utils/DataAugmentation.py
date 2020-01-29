# Data augmentation is a techinique used to generate new traning samples from th eoriginal ones by applying random jitters and perturbations like:
# Tranlations, rotations, changes in scale, shearing or Horizontal and vertical flips
# More advanced techniques can be applied like random perturbations of colos in a given color space and nonlinear geometric distortions.

#To run this script execute:
# python .\VisualizingDataAugmentation.py --image outputs/augmentedImages/batman/batman.jpg --output outputs/augmentedImages/batman
# python .\VisualizingDataAugmentation.py --image outputs/augmentedImages/wolf/wolf.jpg --output outputs/augmentedImages/wolf
import argparse
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class DataAug:
     
     #images should be a numpy array, max is the number of images to generate
     def getImages(self,image, max):
          image = np.expand_dims(image, axis=0)
          #now we can initialize our ImageDataGenerator
          aug = ImageDataGenerator(
              rotation_range=30,  #degree range of random rotation
               width_shift_range=0.1, #horizontal shift
               height_shift_range=0.1, #vertical shift
               shear_range=0.2,   #controls the angel in counterclockwise direction as radians in which our image will allowed to be sheard
               zoom_range=0.2, # a floation point value that allows the image to be zoomed(in,out) according to the following uniform distribution of values -> [1- zoom_range, 1 + zoom_range]
               horizontal_flip=True, #allows the image to be fliped horizontal
               fill_mode="nearest")

          imageGen = aug.flow(image, batch_size=1)
          augmentedImages =[]
          for i in range(max):
               #imageGen.next() generates an output (1,32,32,3)
               #I only want the image (32,32,3) (height, width, depth)
               augmentedImages.append(imageGen.next()[0])
               
          return np.array(augmentedImages)
    
