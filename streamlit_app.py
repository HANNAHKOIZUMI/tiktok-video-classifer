import streamlit as st
import speech_recognition as sr
import wave
import contextlib
import math
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import AudioFileClip
import tempfile

class VideoTranscriber:
    def __init__(self):
        self.data = []

    def extract_audio(self, video_file, audio_file):
        # Extract audio from the video using moviepy
        audioclip = AudioFileClip(video_file)
        audioclip.write_audiofile(audio_file)

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

# Initialize the Streamlit app
def main():
    st.title("TikTok Video Classifier")
    st.write("Upload a video file.")

    # File uploader for videos
    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "mov", "avi"], accept_multiple_files=True)

    if uploaded_files:
        # Initialize the transcriber
        transcriber = VideoTranscriber()

        # Process each uploaded video file
        for idx, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                temp_video_path = temp_video_file.name

            # Process the video file
            result = transcriber.process_video(temp_video_path, idx)

            # Display the transcription results
            st.write(f"**Video File:** {result['Video File']}")
            st.write(f"**Length (seconds):** {result['Length (seconds)']}")
            st.write("**Transcription:**")
            st.write(result['Transcription'])

            # Clean up temporary video file
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()