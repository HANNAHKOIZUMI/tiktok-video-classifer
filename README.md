# TikTok Video Classifier

# Video Processing Pipeline

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

The transcription process uses Amazon Transcribe to convert video files stored in an S3 bucket into text.

### Code Example

```python
import boto3

def transcribe_video(bucket_name, video_file, job_name, region='us-east-1'):
    transcribe = boto3.client('transcribe', region_name=region)
    job_uri = f's3://{bucket_name}/{video_file}'
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='mp4',
        LanguageCode='en-US'
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Waiting for transcription to complete...")
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        return transcript_url

# Example usage
bucket_name = 'your-bucket-name'
video_file = 'your-video.mp4'
job_name = 'your-job-name'
transcript_url = transcribe_video(bucket_name, video_file, job_name)
