import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
from keras import layers
from keras.models import load_model
from tensorflow.keras.models import model_from_json


model_json_file = 'model.json'
model_weights_file = 'model_weights.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)

faceNet = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(0)
import copy

while True:
    ret, frame = cap.read()
    image = copy.deepcopy(frame)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(face, (48,48))
            roi = roi[np.newaxis, :, :, np.newaxis]
            pred = loaded_model.predict(roi)
            text_idx = np.argmax(pred)
            text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            text = text_list[text_idx]
            cv2.putText(image, text, (startX, startY-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
            image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
