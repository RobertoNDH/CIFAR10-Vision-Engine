import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

CIFAR10_CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Dog", "Deer", "Frog", "Horse", "Ship", "Truck"]

@st.cache_resource(show_spinner=False)
def load_cifar10_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_cifar10_final.keras")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Model load error: {e}")

def preprocess_image(image: Image.Image):
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image).astype('float32') / 255.0
        img_array = img_array.reshape((1, 32, 32, 3))
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image processing error: {e}")

def predict_image(model, img_array):
    try:
        predictions = model(img_array, training=False)
        return predictions.numpy()[0]
    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")

def get_class_name(index):
    return CIFAR10_CLASSES[index]
