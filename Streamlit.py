import streamlit as st
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load evaluation metrics from pickle file
with open('evaluation_metrics.pkl', 'rb') as f:
    metrics_dict = pickle.load(f)

st.title("Video Frame Analysis")

st.write("### Evaluation Metrics")
st.write(f"Accuracy: {metrics_dict['accuracy']:.4f}")
st.write(f"Precision: {metrics_dict['precision']:.4f}")
st.write(f"Recall: {metrics_dict['recall']:.4f}")
st.write(f"F1 Score: {metrics_dict['f1_score']:.4f}")
st.write(f"AUC-ROC: {metrics_dict['auc_roc']:.4f}")
st.write(f"Matthews Correlation Coefficient: {metrics_dict['mcc']:.4f}")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])

if uploaded_file is not None:
    
   # Process the uploaded video
   cap = cv2.VideoCapture(uploaded_file.name)  
   frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   st.write(f'Extracted {frames_count} frames from the video.')

   # Here you would load your pre-trained model weights.
   # model_path should be the path where you save your trained GRU model.
   model_path='path_to_your_trained_model.h5'
   gru_model=load_model(model_path)  

   # Make predictions on the extracted features from the uploaded video.
   features_for_prediction=[] # Extract features for prediction here based on your logic
   
   # Assuming you have a function to extract features from uploaded video frames.
   for _ in range(frames_count): 
       # Replace this with actual feature extraction logic based on your trained model.
       pass

   predictions=gru_model.predict(np.array(features_for_prediction))

   # Display predictions or any other relevant information.
   st.write("Predictions made on uploaded video.")
