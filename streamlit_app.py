import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import tempfile
import os
import speech_recognition as sr
import wave
import contextlib
import math
from moviepy.editor import AudioFileClip
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Define the video transcriber class
class VideoTranscriber:
    def __init__(self):
        self.data = []

    def extract_audio(self, video_file, audio_file):
        # Extract audio from the video using moviepy
        audioclip = AudioFileClip(video_file)
        audioclip.write_audiofile(audio_file, logger=None)  # Disable logging to avoid cluttering the output

    def get_audio_duration(self, audio_file):
        # Calculate the duration of the audio file
        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return duration

    def transcribe_audio(self, audio_file):
        # Transcribe the audio using Google Speech Recognition
        r = sr.Recognizer()
        duration = self.get_audio_duration(audio_file)
        total_duration = math.ceil(duration / 60)
        transcription = ""

        for i in range(total_duration):
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source, offset=i*60, duration=60)
            try:
                transcription_segment = r.recognize_google(audio)
                transcription += transcription_segment + " "
            except sr.UnknownValueError:
                print(f"Could not understand audio at segment {i}.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

        return transcription.strip(), duration

    def process_video(self, video_file, idx):
        # Process a single video file to extract and transcribe audio
        audio_file = f"transcribed_speech_{idx}.wav"
        self.extract_audio(video_file, audio_file)
        transcription, duration = self.transcribe_audio(audio_file)
        os.remove(audio_file)  # Clean up temporary audio file
        return {
            'Index': idx,
            'Video File': video_file,
            'Length (seconds)': duration,
            'Transcription': transcription
        }

    def process_videos(self, video_files):
        # Process multiple video files concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_video, video_file, idx) for idx, video_file in enumerate(video_files)]
            for future in as_completed(futures):
                result = future.result()
                self.data.append(result)

    def save_to_dataframe(self):
        # Save results to a DataFrame
        return pd.DataFrame(self.data)

# Define a PyTorch dataset class for sentiment analysis
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        encoded_text = {key: value.squeeze(0) for key, value in encoded_text.items()}
        return encoded_text

# Define a function to process texts in batches
def process_in_batches(texts, model, tokenizer, batch_size, device):
    dataset = SentimentDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            all_outputs.append(outputs.logits.cpu().numpy())

    return np.concatenate(all_outputs, axis=0)

# Initialize the Streamlit app
def main():
    st.title("Video Transcription and Sentiment Analysis App")
    st.write("Upload a BERT model file (.pkl) and video files to perform sentiment analysis on the transcriptions.")

    # File uploader for the BERT model
    uploaded_model_file = st.file_uploader("Upload a BERT model (.pkl) file", type=["pkl"])

    # File uploader for videos
    uploaded_videos = st.file_uploader("Choose video files", type=["mp4", "mov", "avi"], accept_multiple_files=True)

    if uploaded_model_file and uploaded_videos:
        # Save the uploaded model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_model_file:
            temp_model_file.write(uploaded_model_file.read())
            model_path = temp_model_file.name

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Load the trained model state from the uploaded file
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Initialize the transcriber
        transcriber = VideoTranscriber()

        # Process each uploaded video file
        for idx, uploaded_video in enumerate(uploaded_videos):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
                temp_video_file.write(uploaded_video.read())
                temp_video_path = temp_video_file.name

            # Process the video file
            result = transcriber.process_video(temp_video_path, idx)

            # Perform sentiment analysis on the transcription
            transcription = result['Transcription']
            if transcription:
                st.write("Transcription obtained, analyzing sentiment...")
                sentiment_outputs = process_in_batches([transcription], model, tokenizer, batch_size=1, device=device)
                predicted_label = np.argmax(sentiment_outputs, axis=1)[0]
                sentiment_classes = ["Negative", "Neutral", "Positive"]
                sentiment = sentiment_classes[predicted_label]

                # Display the transcription and sentiment results
                st.write(f"**Video File:** {result['Video File']}")
                st.write(f"**Length (seconds):** {result['Length (seconds)']}")
                st.write("**Transcription:**")
                st.write(result['Transcription'])
                st.write(f"**Predicted Sentiment:** {sentiment} (Class {predicted_label})")

            # Clean up temporary video file
            os.remove(temp_video_path)

        # Clean up the temporary model file
        os.remove(model_path)

if __name__ == "__main__":
    main()