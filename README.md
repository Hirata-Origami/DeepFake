# Deepfake with GRU and Streamlit

This repository contains code for deepfake using a GRU-based neural network model deployed with a Streamlit web application. The model classifies videos as either `Real` or `Fake`, leveraging Xception for feature extraction and GRU for sequence analysis.

## Features
- Preprocess video files and extract frames dynamically.
- Feature extraction using the pre-trained Xception model.
- GRU-based deep learning model for temporal sequence analysis.
- Interactive deployment using Streamlit.
- Visualizations for evaluation, including:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve

## Live Demo
Check out the live app [here](https://bharathpranav.streamlit.app/).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/video-classification-gru.git
   cd video-classification-gru
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Dataset

The application processes video files located in specified directories for real and fake videos. You can update the paths to your dataset in the code:

- Real videos directory: `/kaggle/input/faceforensics/FF++/real`
- Fake videos directory: `/kaggle/input/faceforensics/FF++/fake`

## Model Architecture

1. **Feature Extraction**:
   - Uses the pre-trained Xception model, excluding the top layer, to extract frame-level features.
2. **Sequence Analysis**:
   - GRU layers analyze the temporal sequence of frame features.
3. **Output**:
   - A Dense layer with a sigmoid activation outputs the classification result.

## Evaluation Metrics

The model evaluates its performance using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Visualizations
- **Confusion Matrix**: Displays actual vs. predicted labels.
- **Precision-Recall Curve**: Shows the trade-off between precision and recall.

## Usage

1. Upload a video file using the Streamlit interface.
2. The app processes the video, extracts frames, and performs inference.
3. View the classification result along with evaluation metrics and visualizations.

## File Structure

```
├── app.py                 # Streamlit application code
├── model_training.py      # GRU model training and evaluation
├── utils.py               # Utility functions for preprocessing and feature extraction
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
```

## Example Output

### Confusion Matrix
![Confusion Matrix](example_images/confusion_matrix.png)

### Precision-Recall Curve
![Precision-Recall Curve](example_images/precision_recall_curve.png)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any inquiries, feel free to reach out:
- Author: Bharath Pranav
- Email: bharathpranav91@gmail.com
- Live Demo: [Streamlit App](https://bharathpranav.streamlit.app/)

