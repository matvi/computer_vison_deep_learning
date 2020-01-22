from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential

class Shallownet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        shape = (width, height, depth)

        model.add(Conv2D(32, (3,3), padding="same", input_shape=shape))
        model.add(Activation("relu"))
        
        #softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model