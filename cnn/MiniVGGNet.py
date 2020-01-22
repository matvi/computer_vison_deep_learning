#VGG network
#Introudced by Simonyan and Zisserman in 2014
#The primary contribution of their work was demostating 
# that an architecture with very small 3x3 kernel can be trained to increasingly higher depths
# All layer uses 3x3 kernels

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class MiniVGG:

    def build(width, height, depth, classes):
        
        inputShape = (height,width, depth)
        chanDim = -1

        if(K.image_data_format()== 'channels_first'):
            inputShape = (depth, height, width)
            chanDim = 1

        #first layer
        model = Sequential()
        model.add(Conv2D(32, (3,3), input_shape = inputShape ,padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        #second layer
        model.add(Conv2D(32, (3,3), padding='same' ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        #regularization
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        #thirt layer
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        #fourth layer
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
        model.add(Dropout(0.25))

        #classification layer
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model