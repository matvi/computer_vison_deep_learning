This repo introduces a lot of concepts on Convolutional Neural Network.
As I advance I create new files that will be explained here.

The structure of the folders is important. More for the datasets where the labels are obtained for position.
Folders:
    cnn -> Contains the CNN architectures
    datasets -> Contains the images to train the cnn. [animals, flowers17, Smiles]
    outputs -> 
        -> augmentedImages : here I saved augmented Images produces by the script VisualizingDataAugmentation.py
        -> hdf5 files are serialized of trained models each one specifies the cnn used and the dataset used
    processors-> This files are used to process the images when training the data. 

    backpropagation-> Containts examples of backpropagation
        perceptron, perceptronback_gradientDescent and perceptronback_sgd are simple files that I used to understand
        how backpropagation and SGD worked training a simple perceptron to aproximate the funcion XOR.
        For more info -> https://github.com/matvi/backpropagation
    
    utils -> contains common utilities.
        Scripts:
        VisualizingDataAugmentation -> this file is used to take an image an produce augmented images. The results are stored in the outputs/augmentedImages
        VisualizeArhcitecture -> this file is used to visualize the CNN architecture generating an image in the output. As an example you can open the MiniVGG_architecture.png
        DataAugmentation-> this scripts is used to generate augmented images given a images as input. In the flowers17VGG_predictor_unsingData is used to generate new images to predict what the image is using unsined images.
        inspect_model -> used to print the layers in a ccn architecture


    The rest of the files are used to train and test the model.
    For example:

        cifar10ShallowNetSafeModel: this file will train a cnn using the dataset cifar10 in the ShallowNet cnn architecture and will produce a serialized model in the output.
        cifar10ShallowNetPredictor: this file will use the serialized model of the cifar10SahllowNetSafeModel to predict images of the cifar10.

        SmileVGG_modelCreator : this file will train a cnn using the dataset Smiles in the SmallVGGNet cnn architecture and will produce a serialized model in the output folder. (MiniVGGSmiles_model.hdf5)
        SmileVGG_predictor:  this file will use the serialized model of the SmileVGG_modelCreator (MiniVGGSmiles_model.hdf5) to predict when a person is smilling or not using a camera.

        AnimalsShallowNet: this file will train a cnn using the dataset Animals in the ShallowNet cnn architecture and will produce a serialized model in the output. (AnimalsShalloNet_model.hdf5)
        AnimalsShallowNetPredictor: this file will use the serialized model of the AnimalsShallowNet to predict images of Animals dataset using the model  (AnimalsShalloNet_model.hdf5)

        AnimalsVGG_SafeModel: this file will train a cnn using the dataset Animals in the MiniVGGNet cnn architecture and will produce a serialized model in the output. (AnimalsMiniVGG_model.hdf5)
        AnimalsVGG_predictor: this file will use the serialized model of the AnimalsVGG_SafeModel to predict images of Animals dataset using the model  (AnimalsMiniVGG_model.hdf5)

        Flowers17VGG_SafeModel.py : this file will train a cnn using the dataset Flowers17 in the MiniVGGNet cnn architecture and will produce a serialized model in the output called (AnimalsMiniFlowers17.hdf5)
        Flowers17VGG_predictor: this file will use the serialized model of the Flowers17VGG_SafeModel to predict images of Animals dataset using the model  -> (AnimalsMiniFlowers17.hdf5)
        Flowers17VGG_predictor_unsingData: same as Flowers17VGG_predictor but it will generate new images using DataAugmentation script to test the model.

        FineTunningVGG16.py : this explains FineTunning which is a transfer learning technique where que use a pretrained model like VGG16 to train it in our own dataset to obtain higher accuracy.

        