import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Project files
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_cnn_model.keras"
METADATA_PATH = ROOT / "model_metadata.json"

st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

# Page styling
st.markdown("""
<style>
.stApp {
    background-color: #10131c;
    color: white;
}
h1, h2, h3 {
    color: #ff4b6e;
}
.stButton > button {
    background-color: #e11d48;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    """Read class labels from model_metadata.json when available."""
    if not METADATA_PATH.exists():
        return None

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    # Common metadata formats
    for key in ["class_names", "classes", "class_labels"]:
        value = metadata.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]

    for key in ["class_indices", "class_to_idx", "class_to_index"]:
        value = metadata.get(key)
        if isinstance(value, dict):
            return [
                name for name, index in sorted(
                    value.items(), key=lambda item: item[1]
                )
            ]

    return None


st.title("🔬 Skin Cancer Image Classification")
st.write(
    "Upload a skin image to see the prediction made by the trained "
    "Convolutional Neural Network (CNN) model."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload best_cnn_model.keras to GitHub.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the CNN model: {error}")
    st.stop()

class_names = load_class_names()

if class_names is None:
    st.warning(
        "Class names were not found in model_metadata.json. "
        "Please check your metadata file before relying on prediction labels."
    )

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict Image"):
        try:
            # Get the image dimensions expected by the CNN
            input_shape = model.input_shape
            height = input_shape[1]
            width = input_shape[2]

            if height is None or width is None:
                raise ValueError("The model input image size is not defined.")

            resized_image = image.resize((width, height))

            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # The training model should contain the same preprocessing
            # layers used during training.
            prediction = model.predict(image_array, verbose=0)
            probabilities = prediction[0]

            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index]) * 100

            if class_names and predicted_index < len(class_names):
                predicted_class = class_names[predicted_index]
            else:
                predicted_class = f"Class {predicted_index}"

            st.subheader("Prediction Result")
            st.success(f"Predicted class: {predicted_class}")
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Class probabilities")

            if class_names and len(class_names) == len(probabilities):
                labels = class_names
            else:
                labels = [
                    f"Class {i}" for i in range(len(probabilities))
                ]

            for label, probability in zip(labels, probabilities):
                st.write(f"{label}: {probability * 100:.2f}%")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Educational project only. This CNN prediction is not a medical "
    "diagnosis. Consult a qualified healthcare professional about "
    "any concerning skin changes."
)
