import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page Config
st.set_page_config(page_title="Activity Recognition", page_icon="🤖", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Change In Human Activity Recognition</h1>", unsafe_allow_html=True)
st.divider()

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("activity_recogintion_model.h5")

# Load Class Labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Preprocessing Function
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Upload Section
st.subheader("📸 Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Show Image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction
        model = load_model()
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        predicted_class = class_labels[np.argmax(prediction)]

        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>🏷️ Predicted Activity</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>✅ {predicted_class}</h2>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        





#  streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
