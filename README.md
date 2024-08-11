# TikTok Video Classifier

This repository contains a pipeline for processing videos stored in an Amazon S3 bucket. The pipeline includes transcribing videos using Amazon Transcribe, cleaning the transcriptions, running the cleaned data through a DistilBERT model for classification, and using a hybrid model combining text features with a TensorFlow model. Additionally, a Streamlit app is provided for processing and classifying individual video files.

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

The transcription process? uses Amazon Transcribe to convert video files stored in an S3 bucket into text. For details, see [Bulk_Transcribe_AWS.ipynb](scripts/Bulk_Transcribe_AWS.ipynb).

## Cleaning Transcriptions

Once transcriptions are completed, they are cleaned and organized into a DataFrame. See [clean_transcriptions.py](scripts/clean_transcriptions.py) for implementation details.

## Running DistilBERT Model

The transcriptions are fed into a DistilBERT model for classification. See [distilbert_classification.py](scripts/distilbert_classification.py) for the code.

## Hybrid Model Processing

The text features are further processed with a hybrid model combining the text features with a TensorFlow-based model for enhanced predictions. Refer to [hybrid_model.py](scripts/hybrid_model.py) for more information.

## Streamlit App

A Streamlit app allows for uploading a video and receiving a class prediction based on the trained models. See [app.py](app.py) for the full implementation.

## Requirements

- Python 3.x
- Boto3
- Torch
- Transformers
- TensorFlow
- OpenCV
- Pandas
- Streamlit

## Installation

Install the required packages using pip:

```bash
pip install boto3 torch transformers tensorflow opencv-python-headless pandas streamlit


# Example usage
bucket_name = 'your-bucket-name'
video_file = 'your-video.mp4'
job_name = 'your-job-name'
transcript_url = transcribe_video(bucket_name, video_file, job_name)
