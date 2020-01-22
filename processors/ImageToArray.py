from keras.preprocessing.image import img_to_array

# This calss properly orders the channels based on our image_data_format setting
class ImageToArray:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # data Format = none idicates that setting insede keras.json will be used.
        # other dataFormat could be channels_first or channels_last string.
        return img_to_array(image, data_format=self.dataFormat)