from keras.applications import VGG16

model = VGG16(weights="imagenet", include_top=True)

for (i, layer) in enumerate(model.layers):
    print("[INFO] {} \t {}".format(i, layer.__class__.__name__))