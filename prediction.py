import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import urllib.request
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from io import BytesIO 


#best_model = load_model("C:/Users/Lenovo ThinkPad E15/OneDrive - Ryaktive Software Development/Documents/sunglasses-color-classification/best_model.pt")
best_model = tf.keras.models.load_model("C:/Users/Lenovo ThinkPad E15/OneDrive - Ryaktive Software Development/Documents/sunglasses-color-classification/best_model.pt")


class_names_processed = ['black',
'blue',
'brown',
'burgundy',
'clear',
'gold',
'green',
'grey',
'orange',
'pink',
'purple',
'rainbow',
'red', 
'silver', 
'yellow']

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

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
    return predicted_class
        
