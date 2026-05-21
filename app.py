import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

MODEL_PATHS = {
    "ResNet50V2": "models/resnet50v2.keras",
    "MobileNetV3Large": "models/mobilenetv3large.keras",
    "EfficientNetV2S": "models/efficientnetv2s.keras",
    "ConvNeXtTiny": "models/convnexttiny.keras"
}

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape[1:3]
    return model, input_shape

st.title("Fruit & Vegetable Disease Classifier")
st.write("Upload an image and choose a model.")

selected_model = st.selectbox("Choose model", list(MODEL_PATHS.keys()))
model_path = MODEL_PATHS[selected_model]

if not os.path.exists(model_path):
    st.error(f"Model file not found: `{model_path}`")
    st.stop()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model, input_shape = load_model(model_path)
    img = image.resize(input_shape)
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)

    pred_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    predicted_class = CLASS_NAMES[pred_index] if pred_index < NUM_CLASSES else "Unknown"

    st.subheader("Prediction")
    st.write(f"Model: **{selected_model}**")
    st.write(f"Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    st.subheader("Top 3 Predictions")
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    for idx in top3_idx:
        label = CLASS_NAMES[idx] if idx < NUM_CLASSES else f"Class {idx}"
        st.write(f"{label} — {predictions[0][idx]*100:.2f}%")
