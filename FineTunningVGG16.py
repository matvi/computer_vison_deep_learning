#Transfer learning
#Tranfer leaning is a technique used to take an already trained network and use it to trained over a new
#dataset.

#There are two types of transfer learning
#1.- Treat networks as feature extractors where we used a trained network and propagate the inputs
#until a cetain layer -> then we use the output as feature vectors to train a classfier

#2.- Fine-tunning where we cut the Fully connected layers in an already trained network as the VGG16
# and then we introduce a new set of fully connected layers.
# We then train the network on a new dataset without modifying the Feature Maps in the Convolutional Layers.

#Fine-tunning network over VGG16
#To see the network run the utils/inspect_model.py script

import argparse
from imutils import paths
from processors.DataLoader import DataLoader
from processors.AspectAwarePreporcessor import Resize
from processors.ImageToArray import ImageToArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.applications import VGG16
from keras.layers import Input
from cnn.fcLayer import FullyConnected
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

#comments are in base Animals dataset where we have only 3 classes (cat, dog, panda) for easy understanding
# the results are based on training Flowers17 dataset at the end.
# to run this program execute the next script
# python FineTunningVGG16 --dataset C:\\gitProjects\\perceptron\\datasets\\flowers17 --output C:\\gitProjects\\perceptron\\outputs\\Flowers17VGG16.model



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help= "path to the dataset", default="C:\\gitProjects\\perceptron\\datasets\\flowers17")
ap.add_argument("-o","--output", help="output path for the model", default="C:\\gitProjects\\perceptron\\outputs\\Flowers17VGG16.model")
args = vars(ap.parse_args())

print(args["dataset"])

#variables
batchSize = 32
im_size = 224
lr = 0.001
fullyConnectedNodes = 256
epochs25=25
epochs100=100
#get the imagesPath
dataset = args["dataset"]
imagesPath = list(paths.list_images(dataset))

#create the preprocessors that will be used in the images

p_resize = Resize(im_size,im_size)
p_iarr = ImageToArray()
dataloader = DataLoader(preprocessors=[p_resize, p_iarr])
#get the images as numpy arrays and labels
(images, labels) = dataloader.load(imagesPath)
#convert the images format from uint8 [0,255] to float [0,1]
images = images.astype("float") / 255.0
#we get the classes from labels
classes = np.unique(labels)

#we split our data into training data and test data
#trainX.shape -> (2250, 224, 224, 3)
#testX.shape -> (750, 224, 224, 3)
(trainX, testX, trainY, testY) = train_test_split(images, labels)

#apply one hot encoding to the labels
lb = LabelBinarizer()
#lb.fit_transform uses lb.fit to normalize the data and transform to convert the data to one hot encoding
#trainY.shape -> (2250,3)
#trainY[0] -> array([0,1,0]) meaning that it is a dog
trainY = lb.fit_transform(trainY)
# once the data has been normalize for lb.fit we only need to call lb.transform for the rest of the data
#testY before lb.transform -> testY.shape -> (750,) and -> testY[0] -> 'dog'
#testY after lb.transform -> testY.shape -> (750,3) and -> testY[0] -> array([0,1,0])
testY = lb.transform(testY)

#Call the model
#include_top= False is used to exclude the Fully connected layers in the VGG16 net
#Input()` is used to instantiate a Keras tensor.
#we explicity define the input-tensor to be 224,224,3 ussing channels last like in TensorFlow
vgg = VGG16(weights="imagenet", include_top=False, input_tensor= Input(shape = (im_size, im_size, 3)))

#initialize the new fully-connected layer that will be used by the model 
#->FullyConnected.build(newmodel, number of classes, 256 nodes in the FC layer)
vggTail = FullyConnected.build(vgg, classes.shape[0], fullyConnectedNodes)

#construct a new model using the body of VGG16(vgg.input) as the input and the vgg as the output.
#we are placing the head FC model on top of the base model.
model = Model(inputs=vgg.input, outputs = vggTail)

#Befor trainning the network we need to freeze the previews layers to avoid they will be updated
for layer in vgg.layers:
    layer.trainable = False

#we will use RMprop optimizer with an a small leraning rate of 1e-3
#when fine-tunning a network it is necesary to use a smaller learning rate that the one that was used for training the network
opt = RMSprop(lr = lr)
model.compile(optimizer = opt, loss="categorical_crossentropy", metrics=["accuracy"])

#the next step is to train the network. As we freezed the body of the network we only are going to train the FC layers
#we do this for a few epochs [20-30]
aug = ImageDataGenerator(zoom_range=0.2, rotation_range=30, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)

#the stpes_per_epoch = 2250/32 = 70 for Animals dataset
model.fit_generator(aug.flow(trainX, trainY, batch_size=batchSize), validation_data=(testX,testY), epochs=epochs25, steps_per_epoch=len(trainX)//batchSize)

#evaluate the initialization
predict = model.predict(testX, batch_size=batchSize)
print(classification_report(testY.argmax(axis=1), predict.argmax(axis=1), target_names = classes))

#return (vgg.layers, predict, testY,len(trainX)//batchSize)

#after traning the FC layers we need to unfreeze  15 layers so we can trained them

for layer in vgg.layers[15:]:
    layer.trainable = True

#we will use SGD to train the entire network
sgd = SGD(lr = lr, momentum=0.9, decay= lr/epochs100)

model.compile(loss="categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])
model.fit_generator(aug.flow(trainX, trainY, batch_size=batchSize), validation_data=(testX,testY), epochs = epochs100, steps_per_epoch = len(trainX)//batchSize)
predict = model.predict(testX, batch_size=batchSize)
print(classification_report(testY.argmax(axis=1), predict.argmax(axis=1), target_names= classes))

model.save(args["output"])



#   results in Flowers17
#   it took me 12 hours of trainning in a core i7 8700k 3.7 Ghz with 16Gb of ram
#   The model got a 95% of accuracy.
#    >>> from FineTunningVGG16 import fineTunning
#    Using TensorFlow backend.
#    >>> fineTunning().tune()
#    C:\gitProjects\perceptron\datasets\flowers17
#    WARNING:tensorflow:From C:\Users\dmata\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
#    Instructions for updating:
#    Colocations handled automatically by placer.
#    2020-01-28 17:43:53.797517: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
#    2020-01-28 17:43:53.803224: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 12. Tune using inter_op_parallelism_threads for best performance.
#    WARNING:tensorflow:From C:\Users\dmata\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
#    Instructions for updating:
#    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
#    WARNING:tensorflow:From C:\Users\dmata\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
#    Instructions for updating:
#    Use tf.cast instead.
#    Epoch 1/25
#    31/31 [==============================] - 333s 11s/step - loss: 5.3962 - acc: 0.1321 - val_loss: 2.2704 - val_acc: 0.2824
#    Epoch 2/25
#    31/31 [==============================] - 330s 11s/step - loss: 2.3103 - acc: 0.3060 - val_loss: 1.7394 - val_acc: 0.4735
#    Epoch 3/25
#    31/31 [==============================] - 331s 11s/step - loss: 2.1260 - acc: 0.3776 - val_loss: 1.6669 - val_acc: 0.5618
#    Epoch 4/25
#    31/31 [==============================] - 338s 11s/step - loss: 1.7742 - acc: 0.4594 - val_loss: 1.1721 - val_acc: 0.6588
#    Epoch 5/25
#    31/31 [==============================] - 332s 11s/step - loss: 1.4757 - acc: 0.5309 - val_loss: 1.0425 - val_acc: 0.6471
#    Epoch 6/25
#    31/31 [==============================] - 331s 11s/step - loss: 1.4181 - acc: 0.5577 - val_loss: 1.0618 - val_acc: 0.6324
#    Epoch 7/25
#    31/31 [==============================] - 336s 11s/step - loss: 1.2668 - acc: 0.5968 - val_loss: 0.9722 - val_acc: 0.6824
#    Epoch 8/25
#    31/31 [==============================] - 335s 11s/step - loss: 1.1689 - acc: 0.6214 - val_loss: 0.9634 - val_acc: 0.7118
#    Epoch 9/25
#    31/31 [==============================] - 338s 11s/step - loss: 1.1371 - acc: 0.6433 - val_loss: 0.6974 - val_acc: 0.7912
#    Epoch 10/25
#    31/31 [==============================] - 341s 11s/step - loss: 1.0150 - acc: 0.6927 - val_loss: 0.8545 - val_acc: 0.7353
#    Epoch 11/25
#    31/31 [==============================] - 339s 11s/step - loss: 0.9860 - acc: 0.6956 - val_loss: 0.6437 - val_acc: 0.8118
#    Epoch 12/25
#    31/31 [==============================] - 332s 11s/step - loss: 0.9246 - acc: 0.7033 - val_loss: 0.5840 - val_acc: 0.8147
#    Epoch 13/25
#    31/31 [==============================] - 331s 11s/step - loss: 0.9067 - acc: 0.7208 - val_loss: 0.7048 - val_acc: 0.7618
#    Epoch 14/25
#    31/31 [==============================] - 331s 11s/step - loss: 0.8222 - acc: 0.7341 - val_loss: 0.8008 - val_acc: 0.7941
#    Epoch 15/25
#    31/31 [==============================] - 331s 11s/step - loss: 0.7959 - acc: 0.7574 - val_loss: 0.6161 - val_acc: 0.8176
#    Epoch 16/25
#    31/31 [==============================] - 332s 11s/step - loss: 0.7267 - acc: 0.7764 - val_loss: 0.5116 - val_acc: 0.8382
#    Epoch 17/25
#    31/31 [==============================] - 332s 11s/step - loss: 0.7549 - acc: 0.7516 - val_loss: 0.5927 - val_acc: 0.7971
#    Epoch 18/25
#    31/31 [==============================] - 332s 11s/step - loss: 0.7801 - acc: 0.7559 - val_loss: 0.5464 - val_acc: 0.8353
#    Epoch 19/25
#    31/31 [==============================] - 338s 11s/step - loss: 0.6955 - acc: 0.7745 - val_loss: 0.5833 - val_acc: 0.8382
#    Epoch 20/25
#    31/31 [==============================] - 337s 11s/step - loss: 0.6705 - acc: 0.7848 - val_loss: 0.6440 - val_acc: 0.8176
#    Epoch 21/25
#    31/31 [==============================] - 334s 11s/step - loss: 0.6641 - acc: 0.7873 - val_loss: 0.4904 - val_acc: 0.8529
#    Epoch 22/25
#    31/31 [==============================] - 335s 11s/step - loss: 0.6644 - acc: 0.7919 - val_loss: 0.5380 - val_acc: 0.8382
#    Epoch 23/25
#    31/31 [==============================] - 333s 11s/step - loss: 0.6151 - acc: 0.7841 - val_loss: 0.4826 - val_acc: 0.8500
#    Epoch 24/25
#    31/31 [==============================] - 334s 11s/step - loss: 0.6549 - acc: 0.7895 - val_loss: 0.6101 - val_acc: 0.8294
#    Epoch 25/25
#    31/31 [==============================] - 335s 11s/step - loss: 0.6343 - acc: 0.8010 - val_loss: 0.6619 - val_acc: 0.8147
#                precision    recall  f1-score   support
#
#        bluebell       1.00      0.42      0.59        19
#    buttercup       1.00      0.89      0.94        18
#    coltsfoot       0.80      0.80      0.80        25
#        cowslip       0.40      0.77      0.53        13
#        crocus       0.80      0.95      0.87        21
#        daffodil       0.70      0.96      0.81        24
#        daisy       1.00      0.89      0.94        18
#    dandelion       1.00      0.60      0.75        20
#    fritillary       1.00      0.80      0.89        20
#            iris       0.82      1.00      0.90        18
#    lilyvalley       0.94      0.74      0.83        23
#        pansy       0.91      1.00      0.95        20
#        snowdrop       0.53      0.94      0.68        17
#    sunflower       1.00      0.95      0.97        19
#    tigerlily       0.83      0.95      0.88        20
#        tulip       0.82      0.41      0.55        22
#    windflower       0.95      0.83      0.88        23
#
#    micro avg       0.81      0.81      0.81       340
#    macro avg       0.85      0.82      0.81       340
#    weighted avg       0.86      0.81      0.81       340
#
#    Epoch 1/100
#    31/31 [==============================] - 369s 12s/step - loss: 0.7595 - acc: 0.7782 - val_loss: 0.5671 - val_acc: 0.8412
#    Epoch 2/100
#    31/31 [==============================] - 366s 12s/step - loss: 0.5823 - acc: 0.8104 - val_loss: 0.5130 - val_acc: 0.8441
#    Epoch 3/100
#    31/31 [==============================] - 361s 12s/step - loss: 0.6141 - acc: 0.8036 - val_loss: 0.5053 - val_acc: 0.8500
#    Epoch 4/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.4956 - acc: 0.8265 - val_loss: 0.4979 - val_acc: 0.8706
#    Epoch 5/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.3748 - acc: 0.8782 - val_loss: 0.3745 - val_acc: 0.8941
#    Epoch 6/100
#    31/31 [==============================] - 359s 12s/step - loss: 0.3077 - acc: 0.9065 - val_loss: 0.3523 - val_acc: 0.9059
#    Epoch 7/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.3026 - acc: 0.8990 - val_loss: 0.4266 - val_acc: 0.8765
#    Epoch 8/100
#    31/31 [==============================] - 359s 12s/step - loss: 0.2773 - acc: 0.9062 - val_loss: 0.3431 - val_acc: 0.9118
#    Epoch 9/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.3273 - acc: 0.8888 - val_loss: 0.5016 - val_acc: 0.8912
#    Epoch 10/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.2554 - acc: 0.9149 - val_loss: 0.4138 - val_acc: 0.8912
#    Epoch 11/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.2271 - acc: 0.9149 - val_loss: 0.3487 - val_acc: 0.9118
#    Epoch 12/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1995 - acc: 0.9384 - val_loss: 0.4383 - val_acc: 0.8853
#    Epoch 13/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.2405 - acc: 0.9241 - val_loss: 0.4522 - val_acc: 0.8794
#    Epoch 14/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1895 - acc: 0.9444 - val_loss: 0.3918 - val_acc: 0.8971
#    Epoch 15/100
#    31/31 [==============================] - 360s 12s/step - loss: 0.2060 - acc: 0.9293 - val_loss: 0.4047 - val_acc: 0.9147
#    Epoch 16/100
#    31/31 [==============================] - 363s 12s/step - loss: 0.1705 - acc: 0.9412 - val_loss: 0.3708 - val_acc: 0.9176
#    Epoch 17/100
#    31/31 [==============================] - 362s 12s/step - loss: 0.1399 - acc: 0.9526 - val_loss: 0.3715 - val_acc: 0.9147
#    Epoch 18/100
#    31/31 [==============================] - 360s 12s/step - loss: 0.1870 - acc: 0.9320 - val_loss: 0.3409 - val_acc: 0.9118
#    Epoch 19/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1455 - acc: 0.9523 - val_loss: 0.3827 - val_acc: 0.9265
#    Epoch 20/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1591 - acc: 0.9412 - val_loss: 0.4692 - val_acc: 0.9088
#    Epoch 21/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1613 - acc: 0.9404 - val_loss: 0.4089 - val_acc: 0.9235
#    Epoch 22/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1562 - acc: 0.9453 - val_loss: 0.3784 - val_acc: 0.9235
#    Epoch 23/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1389 - acc: 0.9444 - val_loss: 0.3534 - val_acc: 0.9088
#    Epoch 24/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.1278 - acc: 0.9554 - val_loss: 0.3428 - val_acc: 0.9235
#    Epoch 25/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1379 - acc: 0.9492 - val_loss: 0.2813 - val_acc: 0.9324
#    Epoch 26/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0978 - acc: 0.9656 - val_loss: 0.3756 - val_acc: 0.9029
#    Epoch 27/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0831 - acc: 0.9738 - val_loss: 0.3153 - val_acc: 0.9324
#    Epoch 28/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1133 - acc: 0.9605 - val_loss: 0.3612 - val_acc: 0.9088
#    Epoch 29/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1171 - acc: 0.9581 - val_loss: 0.2997 - val_acc: 0.9324
#    Epoch 30/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1339 - acc: 0.9585 - val_loss: 0.3944 - val_acc: 0.9294
#    Epoch 31/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1011 - acc: 0.9698 - val_loss: 0.3420 - val_acc: 0.9353
#    Epoch 32/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0780 - acc: 0.9706 - val_loss: 0.4008 - val_acc: 0.9235
#    Epoch 33/100
#    31/31 [==============================] - 359s 12s/step - loss: 0.1057 - acc: 0.9667 - val_loss: 0.3004 - val_acc: 0.9324
#    Epoch 34/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1290 - acc: 0.9604 - val_loss: 0.2820 - val_acc: 0.9294
#    Epoch 35/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1035 - acc: 0.9657 - val_loss: 0.3170 - val_acc: 0.9265
#    Epoch 36/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0984 - acc: 0.9643 - val_loss: 0.3107 - val_acc: 0.9294
#    Epoch 37/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0835 - acc: 0.9687 - val_loss: 0.2549 - val_acc: 0.9382
#    Epoch 38/100
#    31/31 [==============================] - 359s 12s/step - loss: 0.0654 - acc: 0.9829 - val_loss: 0.2919 - val_acc: 0.9353
#    Epoch 39/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0999 - acc: 0.9646 - val_loss: 0.2590 - val_acc: 0.9471
#    Epoch 40/100
#    31/31 [==============================] - 359s 12s/step - loss: 0.0618 - acc: 0.9829 - val_loss: 0.3151 - val_acc: 0.9324
#    Epoch 41/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0581 - acc: 0.9808 - val_loss: 0.2943 - val_acc: 0.9412
#    Epoch 42/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0710 - acc: 0.9745 - val_loss: 0.2727 - val_acc: 0.9382
#    Epoch 43/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0813 - acc: 0.9667 - val_loss: 0.3525 - val_acc: 0.9353
#    Epoch 44/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0598 - acc: 0.9819 - val_loss: 0.4062 - val_acc: 0.9382
#    Epoch 45/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0526 - acc: 0.9857 - val_loss: 0.3447 - val_acc: 0.9324
#    Epoch 46/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0635 - acc: 0.9798 - val_loss: 0.2725 - val_acc: 0.9324
#    Epoch 47/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0500 - acc: 0.9849 - val_loss: 0.3084 - val_acc: 0.9353
#    Epoch 48/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0642 - acc: 0.9757 - val_loss: 0.3199 - val_acc: 0.9382
#    Epoch 49/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0766 - acc: 0.9748 - val_loss: 0.3085 - val_acc: 0.9412
#    Epoch 50/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.1069 - acc: 0.9685 - val_loss: 0.2272 - val_acc: 0.9382
#    Epoch 51/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0590 - acc: 0.9819 - val_loss: 0.1994 - val_acc: 0.9471
#    Epoch 52/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0553 - acc: 0.9817 - val_loss: 0.2510 - val_acc: 0.9441
#    Epoch 53/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0497 - acc: 0.9829 - val_loss: 0.2607 - val_acc: 0.9412
#    Epoch 54/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0580 - acc: 0.9808 - val_loss: 0.3506 - val_acc: 0.9294
#    Epoch 55/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0754 - acc: 0.9748 - val_loss: 0.2838 - val_acc: 0.9235
#    Epoch 56/100
#    31/31 [==============================] - 359s 12s/step - loss: 0.0678 - acc: 0.9797 - val_loss: 0.3125 - val_acc: 0.9324
#    Epoch 57/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0591 - acc: 0.9829 - val_loss: 0.2743 - val_acc: 0.9324
#    Epoch 58/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0517 - acc: 0.9849 - val_loss: 0.2688 - val_acc: 0.9353
#    Epoch 59/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0461 - acc: 0.9826 - val_loss: 0.3556 - val_acc: 0.9324
#    Epoch 60/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0388 - acc: 0.9869 - val_loss: 0.3388 - val_acc: 0.9353
#    Epoch 61/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0425 - acc: 0.9869 - val_loss: 0.3380 - val_acc: 0.9353
#    Epoch 62/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0385 - acc: 0.9878 - val_loss: 0.3438 - val_acc: 0.9324
#    Epoch 63/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0718 - acc: 0.9768 - val_loss: 0.3968 - val_acc: 0.9147
#    Epoch 64/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0570 - acc: 0.9829 - val_loss: 0.2755 - val_acc: 0.9353
#    Epoch 65/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0741 - acc: 0.9758 - val_loss: 0.3127 - val_acc: 0.9294
#    Epoch 66/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0492 - acc: 0.9829 - val_loss: 0.3329 - val_acc: 0.9324
#    Epoch 67/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0637 - acc: 0.9847 - val_loss: 0.4931 - val_acc: 0.9206
#    Epoch 68/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0568 - acc: 0.9857 - val_loss: 0.4109 - val_acc: 0.9235
#    Epoch 69/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0616 - acc: 0.9778 - val_loss: 0.2986 - val_acc: 0.9441
#    Epoch 70/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0540 - acc: 0.9879 - val_loss: 0.4763 - val_acc: 0.9147
#    Epoch 71/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0548 - acc: 0.9819 - val_loss: 0.3691 - val_acc: 0.9265
#    Epoch 72/100
#    31/31 [==============================] - 363s 12s/step - loss: 0.0561 - acc: 0.9757 - val_loss: 0.3710 - val_acc: 0.9382
#    Epoch 73/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0467 - acc: 0.9839 - val_loss: 0.3404 - val_acc: 0.9324
#    Epoch 74/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0417 - acc: 0.9839 - val_loss: 0.3606 - val_acc: 0.9382
#    Epoch 75/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0359 - acc: 0.9879 - val_loss: 0.3553 - val_acc: 0.9324
#    Epoch 76/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0506 - acc: 0.9808 - val_loss: 0.3154 - val_acc: 0.9471
#    Epoch 77/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0410 - acc: 0.9849 - val_loss: 0.3654 - val_acc: 0.9353
#    Epoch 78/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0388 - acc: 0.9899 - val_loss: 0.3648 - val_acc: 0.9265
#    Epoch 79/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0263 - acc: 0.9919 - val_loss: 0.3450 - val_acc: 0.9324
#    Epoch 80/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0414 - acc: 0.9889 - val_loss: 0.2574 - val_acc: 0.9382
#    Epoch 81/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0340 - acc: 0.9859 - val_loss: 0.3380 - val_acc: 0.9412
#    Epoch 82/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0406 - acc: 0.9879 - val_loss: 0.3667 - val_acc: 0.9382
#    Epoch 83/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0330 - acc: 0.9855 - val_loss: 0.4506 - val_acc: 0.9235
#    Epoch 84/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0395 - acc: 0.9869 - val_loss: 0.3215 - val_acc: 0.9382
#    Epoch 85/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0422 - acc: 0.9839 - val_loss: 0.3919 - val_acc: 0.9353
#    Epoch 86/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0376 - acc: 0.9857 - val_loss: 0.3228 - val_acc: 0.9471
#    Epoch 87/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0291 - acc: 0.9928 - val_loss: 0.3323 - val_acc: 0.9471
#    Epoch 88/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0337 - acc: 0.9889 - val_loss: 0.4917 - val_acc: 0.9206
#    Epoch 89/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0480 - acc: 0.9827 - val_loss: 0.3976 - val_acc: 0.9324
#    Epoch 90/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0417 - acc: 0.9849 - val_loss: 0.3417 - val_acc: 0.9324
#    Epoch 91/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0398 - acc: 0.9889 - val_loss: 0.2995 - val_acc: 0.9382
#    Epoch 92/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0244 - acc: 0.9899 - val_loss: 0.3502 - val_acc: 0.9353
#    Epoch 93/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0468 - acc: 0.9819 - val_loss: 0.3955 - val_acc: 0.9412
#    Epoch 94/100
#    31/31 [==============================] - 358s 12s/step - loss: 0.0440 - acc: 0.9829 - val_loss: 0.2958 - val_acc: 0.9382
#    Epoch 95/100
#    31/31 [==============================] - 357s 12s/step - loss: 0.0359 - acc: 0.9849 - val_loss: 0.3994 - val_acc: 0.9235
#    Epoch 96/100
#    31/31 [==============================] - 371s 12s/step - loss: 0.0284 - acc: 0.9950 - val_loss: 0.3266 - val_acc: 0.9412
#    Epoch 97/100
#    31/31 [==============================] - 372s 12s/step - loss: 0.0280 - acc: 0.9909 - val_loss: 0.4097 - val_acc: 0.9353
#    Epoch 98/100
#    31/31 [==============================] - 371s 12s/step - loss: 0.0491 - acc: 0.9839 - val_loss: 0.3118 - val_acc: 0.9471
#    Epoch 99/100
#    31/31 [==============================] - 361s 12s/step - loss: 0.0441 - acc: 0.9869 - val_loss: 0.4102 - val_acc: 0.9294
#    Epoch 100/100
#    31/31 [==============================] - 361s 12s/step - loss: 0.0362 - acc: 0.9889 - val_loss: 0.3763 - val_acc: 0.9353
#                precision    recall  f1-score   support
#
#        bluebell       0.89      0.84      0.86        19
#    buttercup       1.00      1.00      1.00        18
#    coltsfoot       0.89      0.96      0.92        25
#        cowslip       0.81      1.00      0.90        13
#        crocus       0.91      0.95      0.93        21
#        daffodil       0.96      1.00      0.98        24
#        daisy       1.00      0.94      0.97        18
#    dandelion       1.00      0.80      0.89        20
#    fritillary       1.00      0.85      0.92        20
#            iris       1.00      1.00      1.00        18
#    lilyvalley       1.00      0.91      0.95        23
#        pansy       0.95      1.00      0.98        20
#        snowdrop       0.74      1.00      0.85        17
#    sunflower       0.95      1.00      0.97        19
#    tigerlily       1.00      1.00      1.00        20
#        tulip       0.94      0.73      0.82        22
#    windflower       0.92      0.96      0.94        23
#
#    micro avg       0.94      0.94      0.94       340
#    macro avg       0.94      0.94      0.93       340
#    weighted avg       0.94      0.94      0.93       340
#
#
#
#