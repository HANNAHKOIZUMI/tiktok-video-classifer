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
