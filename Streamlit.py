import streamlit as st
import os
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model

# Initialize Streamlit app
def main():
    st.set_page_config(
        page_title="Deepfake Detector",
        page_icon="🕵‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Deepfake Detection App")
    st.markdown(
        "Upload a video file, and the app will analyze it to determine if it's a deepfake."
    )

    uploaded_file = st.file_uploader(
        "Upload a Video File", type=["mp4", "avi", "mkv"]
    )

    if uploaded_file:
        with st.spinner("Processing video..."):
            # Save uploaded video temporarily
            temp_file_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Preprocess and classify
            prediction = process_and_predict(temp_file_path)

            # Display results
            if prediction is not None:
                if prediction > 0.5:
                    st.error("This video is likely a deepfake.")
                else:
                    st.success("This video is likely genuine.")
            else:
                st.warning("Could not process the video. Please try again.")

        # Clean up
        if temp_file_path:
            os.remove(temp_file_path)

# Load the pre-trained GRU model and the Xception feature extractor
@st.cache_resource
def load_models():
    # Load the GRU model from the .h5 file
    gru_model = tf.keras.models.load_model("gru_model.h5")
    
    # Load the Xception model
    xception_model = Xception(weights="imagenet", include_top=False, pooling="avg")
    return gru_model, xception_model


# Preprocess video
def preprocess_video(video_path, max_frames=30, resize_shape=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0 and len(frames) < max_frames:
            frame_resized = cv2.resize(frame, resize_shape)
            frames.append(frame_resized)

        count += 1

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((*resize_shape, 3), dtype=np.uint8))

    return np.array(frames)

# Extract features using Xception
def extract_features(frames, xception_model):
    frames = tf.keras.applications.xception.preprocess_input(frames)
    return xception_model.predict(frames, batch_size=16)

# Process video and predict
def process_and_predict(video_path):
    gru_model, xception_model = load_models()

    try:
        frames = preprocess_video(video_path)
        features = extract_features(frames, xception_model)

        # Reshape for GRU input
        features = np.expand_dims(features, axis=0)

        prediction = gru_model.predict(features)[0][0]
        return prediction

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    main()
