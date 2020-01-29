from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FullyConnected():
    @staticmethod
    #This method requires three parameters: The baseModel (the body of the network), the total number of classes in our dataset, and D the number of nodes in the fully-connected layer.
    def build(baseModel, classes, D):
        #initializes the headModel which is responsible for connecting our network with the rest of the body, baseModel.output.
        headModel = baseModel.output
        #build a simple fullyy-connected architecture
        #this fully-connected head is very simplistic compared to the original head from VGG16 which consits of two sets of 40096 FC layers.
        #Goever, for most fine-tunning problems you are not seeking to replicate the original head of the network, but rather simplify it so it is easier to fine-tune 
        #the few parameters in the head.
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        #softmaxLayer
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel
        
