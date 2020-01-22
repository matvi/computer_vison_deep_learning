
import argparse
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-m","--model", required=True, help = "path to model")
args = vars(ap.parse_args())

#we load the model that was trained for Smiles
model = load_model(args["model"])


#load the face detector published in 2001 by Paul Viola an dMichael Jones.
detector = cv2.CascadeClassifier("C:\gitProjects\perceptron\datasets\SMILEsmileD\SMILEs\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while(True):
    #capture frame-by-frame
    (ret, frame) = cap.read()
    #we set the color to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    copyImg = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)


    for (fX, fY, fW, fH) in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (32,32))
         #add other channels
        #https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
        roi = np.stack((roi,)*3, axis=-1)
        roi = img_to_array(roi)
        roi = roi.astype("float") / 255.0
       
        #we need to add an extra dimension before
        roi = np.expand_dims(roi, axis=0)

        #now we can send to prediction
        print(roi.shape)
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "smiling" if smiling > notSmiling else "not smiling"

        cv2.putText(copyImg, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),2)
        cv2.rectangle(copyImg, (fX,fY), (fX+fW, fY+ fH), (0,0,255),2)


    cv2.imshow('face', copyImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    #display the results
    #cv2.imshow('grame', copyImg)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
cap.release()
cv2.destroyAllWindows()