# This file will be created in /content/utils.py

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# load the saved fine-tuned model
model = load_model("mobilenet_v2_finetuned.h5")

class_labels = ['Angry', 'Happy', 'Neutral', 'fear', 'disgust']

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    top_idx = np.argmax(preds[0])
    return class_labels[top_idx], float(preds[0][top_idx]), preds[0].tolist()
