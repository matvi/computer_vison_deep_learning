from keras.datasets import cifar10
import argparse
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from cnn.ShallowNet import Shallownet
import matplotlib.pyplot as plt

# to run this example execute:
# python .\cifar10ShallowNetSafeModel.py --output_model_dir outputs/cifar10ShallowNet_model.hdf5 --output_plot_dir outputs/cifar10ShallowNet_plot.png
def printInfo(info):
    print("[INFO] " + info)


#this class will train the cifar10 using the most basic CNN called ShallowNet and will produce the model in a hi5 format so it can be used for testing.

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output_model_dir", required = True, help="output direction for the model")
ap.add_argument("-i","--output_plot_dir",  required = True, help="output direction for the training histogram")
args = vars(ap.parse_args())

#the cifar10 is formed by 10 digits [0-9] 
#Each class is formed by a 32*32*3 pixel
#Each class has 50,000 for training and 10,000 for testing
printInfo("Loading cifar10")
((trainX, trainY ), (testX, testY)) = cifar10.load_data()


# First transform our data from int8 to floating 
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#we then change the images to one dimension so we can feed our neural network
#that is the size of the dataset, the size of the image(32*32*3 = 3072)

#trainX = trainX.reshape(trainX.shape[0], 3072)
#testX = testX.reshape(testX.shape[0], 3072)

#we need to transform our trainY vector to a one-hot-encoding matrix
lb = LabelBinarizer()
#fit will find the media and standar desviation and then it applies the transform with those values
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

###we´ll create the model 


#load the ShallowNet with the image configuration (32*32*3) and the number of classes 10
#sh = ShallowNet.Shallownet()
model = Shallownet.build(width=32, height=32, depth=3, classes=10)

# using SGD with a learning rate = 0.01
opt = SGD(lr=0.01)
#we compile the model with the parameters to train the network
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#we train the network
printInfo("Training the model")
epochs=100
H = model.fit(trainX, trainY, epochs=epochs, batch_size = 32, validation_data=(testX, testY))

#print the training results
printInfo("Evaluating the network")
predictions = model.predict(testX, batch_size = 32)
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names = labels))

#saving the model
#this process is called model serialization
printInfo("Serializing the network in : " + args["output_model_dir"])
model.save(args["output_model_dir"])

#creating the training plot
plt.style.use("ggplot")
plt.figure()
plt.plot(range(0,epochs), H.history["loss"], label="training_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss/ Acurracy")
plt.legend()

printInfo("Saving the plot image in : " + args["output_plot_dir"])
plt.savefig(args["output_plot_dir"])





