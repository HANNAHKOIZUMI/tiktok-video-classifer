# TikTok Video Classifier
### Overview
This repository contains a pipeline for processing videos stored in an Amazon S3 bucket. The pipeline includes transcribing videos using Amazon Transcribe, cleaning the transcriptions, running the cleaned data through a DistilBERT model for classification, and using a hybrid model combining text features with a TensorFlow model. Additionally, a Streamlit app is provided for processing and classifying individual video files.

### Goals
1. Classify videos into three categories: neutral (0), anti-Biden (1), pro-Biden (2)
2. Bulk classify videos into categories to save on labor
3. Create a user-friendly interface to demostrate the usage of the classification model on one video. 

### Results
Dataset: 550 videos (balanced between the three categories), manually categorized into one of the three categories to create a training and validation set. Stored in a S3 bucket. 
Distilbert model validation accuracy: 79%
Hybrid model validation accuracy: 78%

### Future Improvements
1. Automate uploading videos to the s3 bucket
2. Improve the confidence scores of the single video classifier connected to streamlit
3. Create and train with a new dataset with Kamala Harris videos

## Table of Contents

1. [Transcribing Videos](#transcribing-videos)
2. [Cleaning Transcriptions](#cleaning-transcriptions)
3. [Running DistilBERT Model](#running-distilbert-model)
4. [Hybrid Model Processing](#hybrid-model-processing)
5. [Streamlit App](#streamlit-app)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Usage](#usage)

## Transcribing Videos

The transcription process uses Amazon Transcribe to convert video files stored in an S3 bucket and stores the transcriptions in a separate S3 bucket. The code also exports the transcriptions to a dataframe and an excel. See [Bulk_Transcribe_AWS.ipynb](Bulk_Transcribe_AWS.ipynb) for the code.

## Cleaning Transcriptions

This notebook combines the training dataframe with the video transcriptions, it uses the names as an index to join the dataframes. See [Bulk_Clean_Transcriptions.ipynb](Bulk_Transcribe_AWS.ipynb) for implementation details.

## Running DistilBERT Model

The transcriptions are fed into a DistilBERT model for classification. See [Bulk_Clean_Transcriptions.ipynb](Bulk_Clean_Transcriptions.ipynb) for the code.

## Hybrid Model Processing

The text features are further processed with a hybrid model combining the text features with a TensorFlow-based model for enhanced predictions. Refer to [hybrid_model_final.ipynb](hybrid_model_final.ipynb) for more information. The architecture for the model:
![Screenshot 2024-08-11 at 9 40 05 PM](https://github.com/user-attachments/assets/024e586c-488d-4371-a816-5e8c2e8206ff)

## Streamlit App

A Streamlit app allows for uploading a video and receiving a class prediction based on the trained models. See [streamlit_app.py](streamlit_app.py) for the full implementation.

## Requirements

- Python 3.x
- Boto3
- Torch
- Transformers
- TensorFlow
- OpenCV
- Pandas
- Streamlit

# Process a single video

```bash

import cv2
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


N_FRAMES = 3
HEIGHT = 112
WIDTH = 112
MAX_TEXT_FEATURES = 3
model_path = './models/hybrid_model.pth'
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_video(video_path, n_frames):
    """
    Preprocess video by extracting frames and resizing them to the required dimensions.

    Args:
        video_path (str): Path to the video file.
        n_frames (int): Number of frames to extract.

    Returns:
        np.ndarray: Array of processed video frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = frame / 255.0  # Normalize pixel values to [0, 1]
        frames.append(frame)
    cap.release()

    # If not enough frames, pad with zeros
    if len(frames) < n_frames:
        frames.extend([np.zeros((HEIGHT, WIDTH, 3))] * (n_frames - len(frames)))

    return np.array(frames)

def analyze_video(video_path, model, text_features):
    """
    Analyze a video and return the predicted class and confidence.

    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): Pretrained model for classification.
        text_features (str): Additional text features.

    Returns:
        int: Predicted class.
        float: Confidence score of the prediction.
    """
    # Preprocess video
    video_frames = preprocess_video(video_path, N_FRAMES)
    video_frames = np.expand_dims(video_frames, axis=0)  # Add batch dimension

    # Convert video frames to PyTorch tensor
    video_frames_tensor = torch.tensor(video_frames, dtype=torch.float32).permute(0, 4, 1, 2, 3).to(device)  # (N, C, T, H, W)

    # Tokenize text features
    encoded_text = tokenizer(text_features, padding='max_length', truncation=True, max_length=MAX_TEXT_FEATURES, return_tensors='pt')
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()

    predicted_class = np.argmax(logits, axis=-1)[0]
    confidence = logits[0][predicted_class]

    return predicted_class, confidence

predicted_class, confidence = analyze_video(video_path, model, text_features)
print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
