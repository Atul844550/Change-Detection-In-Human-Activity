import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

# File IDs from Google Drive
MODEL_FILE_ID = "1Sbylinsm8dD9q7VpUHNRu9ccO3TiAprp"
JSON_FILE_ID = "1a2TPVlsoIZoua__ao8TbroEeDW1v-Ihl"

MODEL_PATH = "activity_recogintion_model.h5"
JSON_PATH = "class_indices.json"

# Function to download files using gdown
def download_file_from_gdrive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Download model and class labels
@st.cache_resource
def load_resources():
    download_file_from_gdrive(MODEL_FILE_ID, MODEL_PATH)
    download_file_from_gdrive(JSON_FILE_ID, JSON_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(JSON_PATH, "r") as f:
        class_indices = json.load(f)
    
    class_labels = list(class_indices.keys())
    return model, class_labels

# Preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.set_page_config(page_title="Activity Recognition", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Change In Human Activity Recognition</h1>", unsafe_allow_html=True)
st.divider()

# Upload Section
st.subheader("📸 Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model and labels
        model, class_labels = load_resources()

        # Predict
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        predicted_class = class_labels[np.argmax(prediction)]

        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>🏷️ Predicted Activity</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>✅ {predicted_class}</h2>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")


#  streamlit run through_google_drive.py --server.enableCORS false --server.enableXsrfProtection false
