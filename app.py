
import os
import sys
#https://github.com/streamlit/streamlit/issues/511
#pip install --upgrade protobuf
#pip install streamlit

import streamlit as st
import cv2 
import numpy as np
import pandas as pd
from PIL import Image,ImageEnhance

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import time


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
import time
from keras import backend as K

from sklearn.datasets import load_files  
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.utils import np_utils
#import cv2
import keras, os
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D, AveragePooling2D 
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.constraints import max_norm
from numpy import expand_dims
from io import BytesIO
from PIL import Image
#from tabulate import tabulate
import matplotlib.pyplot as plt
#%matplotlib inline



# import the models for further classification experiments
from tensorflow.keras.applications import DenseNet169

import matplotlib.pyplot as plt

# imports for reproducibility
import tensorflow as tf
import random
import os

st.title("Sunglasses Lens Color Image Classification App")
st.write("")

add_selectbox = st.sidebar.selectbox(
    "Search by",
    ("Lens Color", "Frame Color", "Shape")
)

file_up = st.file_uploader("Upload an image", type="jpg")

from PIL import Image
image = Image.open(file_up)
st.image(image, caption='Uploaded Image.', use_column_width=True)


def get_prediction(image):
    image = np.expand_dims(image, axis=0)
    prediction = best_model.predict(image)
    predicted_class = np.argmax(prediction)
    return class_names_processed[predicted_class]

def url_to_image_(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = preprocess_input(cv2.resize(image, dsize=(224,224)))
    return image_resized, image_

def predict_url(url):
    image_resized, image = url_to_image_(url)
    predicted_class = get_prediction(image_resized)
    plt.imshow(image)
    plt.title("Predicted : " + predicted_class)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
        
