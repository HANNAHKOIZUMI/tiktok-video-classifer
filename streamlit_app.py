import streamlit as st
import numpy as np
import tempfile
import os
import speech_recognition as sr
import wave
import contextlib
import math
from moviepy.editor import AudioFileClip
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import einops
import cv2

# Define custom layers
class Conv2Plus1D(layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = Sequential([
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

class Project(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv = layers.Conv3D(filters, kernel_size=1)

    def call(self, x):
        return self.conv(x)

class ResidualMain(layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

class ResizeVideo(layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(images, '(b t) h w c -> b t h w c', t=old_shape['t'])
        return videos

# Define the video transcriber class
class VideoTranscriber:
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

        return transcription.strip()

    def process_video(self, video_file):
        # Process a single video file to extract and transcribe audio
        audio_file = "transcribed_speech.wav"
        self.extract_audio(video_file, audio_file)
        transcription = self.transcribe_audio(audio_file)
        os.remove(audio_file)  # Clean up temporary audio file
        return transcription

# Define the frame processing function
def preprocess_frames(video_path, n_frames, height, width):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))  # Resize frames to specified size
        frame = frame / 255.0  # Normalize
        frames.append(frame)
    cap.release()
    if len(frames) < n_frames:
        frames.extend([np.zeros((height, width, 3))] * (n_frames - len(frames)))  # Pad with zeros
    frames = np.array(frames)
    return frames

# Initialize the Streamlit app
def main():
    st.title("Video Transcription and Sentiment Analysis App")
    st.write("Upload a trained model (.h5) and a video file to perform sentiment analysis on the transcription.")

    # File uploader for the model
    uploaded_model_file = st.file_uploader("Upload a trained model (.h5) file", type=["h5"])

    # File uploader for a video
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_model_file and uploaded_video:
        # Save the uploaded model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_model_file:
            temp_model_file.write(uploaded_model_file.read())
            model_path = temp_model_file.name

        # Load the trained model from the uploaded file
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ResizeVideo": ResizeVideo,
                "Conv2Plus1D": Conv2Plus1D,
                "Project": Project,
                "ResidualMain": ResidualMain
            }
        )

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Initialize the transcriber
        transcriber = VideoTranscriber()

        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            temp_video_path = temp_video_file.name

        # Process the video file
        transcription = transcriber.process_video(temp_video_path)
        st.write("**Transcription:**")
        st.write(transcription)

        # Extract text features using the tokenizer
        encoded_text = tokenizer(transcription, padding='max_length', truncation=True, max_length=128, return_tensors='np')
        text_features = np.array(encoded_text['input_ids']).squeeze()

        # Preprocess video frames
        HEIGHT, WIDTH, N_FRAMES = 112, 112, 3
        video_frames = preprocess_frames(temp_video_path, N_FRAMES, HEIGHT, WIDTH)

        # Reshape text features to match the expected input shape
        text_features = text_features.reshape((1, -1))

        # Make predictions
        predictions = model.predict([np.array([video_frames]), text_features])
        predicted_label = np.argmax(predictions, axis=1)[0]

        # Define sentiment classes
        sentiment_classes = [0,1,2]
        sentiment = sentiment_classes[predicted_label]

        # Display the sentiment results
        st.write(f"**Predicted Sentiment:** {sentiment} (Class {predicted_label})")

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(model_path)

if __name__ == "__main__":
    main()