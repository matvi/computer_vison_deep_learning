from cnn.MiniVGGNet import MiniVGG
from keras.utils import plot_model

model = MiniVGG.build(32,32,3,10)
plot_model(model, to_file="C:\gitProjects\perceptron\outputs\pythonMiniVGG_architecture.png", show_shapes=True)