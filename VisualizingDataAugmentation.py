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

ap = argparse.ArgumentParser()
ap.add_argument("-m","--image",required = True, help = "pathToImage")
ap.add_argument("-o", "--output", required = True, help= "output generated images")
args = vars(ap.parse_args())

pathImage = args["image"]
image = cv2.imread(pathImage)
#we transform the image to the format that keras needs -> that is a numpy array -> for tensorflow it uses channel last
image = img_to_array(image)
#the image has a shape (height, width, depth) we need to add another dimension so we can use it in the keras library (1, height, widht, depth)
print(image.shape)
image = np.expand_dims(image, axis=0)
print(image.shape)
#now we can initialize our ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=30,  #degree range of random rotation
     width_shift_range=0.1, #horizontal shift
     height_shift_range=0.1, #vertical shift
     shear_range=0.2,   #controls the angel in counterclockwise direction as radians in which our image will allowed to be sheard
     zoom_range=0.2, # a floation point value that allows the image to be zoomed(in,out) according to the following uniform distribution of values -> [1- zoom_range, 1 + zoom_range]
     horizontal_flip=True, #allows the image to be fliped horizontal
     fill_mode="nearest")

print("[INFO] Generation Images")

imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"], save_prefix="image", save_format="jpg")

i = 0
for im in imageGen:
    i += 1
    if i ==10:
        break
    
