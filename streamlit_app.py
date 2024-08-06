import streamlit as st
import tempfile
import cv2
import numpy as np
import pickle
from transformers import AutoTokenizer
import whisper

# Streamlit setup
st.title("TikTok Video Classifier")
st.write("To get started, upload a video file below.")

# File uploader for video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Constants
    N_FRAMES = 3
    HEIGHT = 112
    WIDTH = 112
    MAX_TEXT_FEATURES = 128  # Adjusted for reasonable text input size

    # Define the path to the saved model
    model_path = './hybrid_model.pkl'  # Ensure the path is correct

    # Load the model using pickle
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Initialize the tokenizer
    model_name = "distilbert-base-uncased"
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

    def transcribe_video(video_path):
        """
        Transcribe audio from a video using Whisper.

        Args:
            video_path (str): Path to the video file.

        Returns:
            str: Transcribed text.
        """
        model = whisper.load_model("base")  # You can choose "tiny", "base", "small", "medium", or "large"
        result = model.transcribe(video_path)
        return result['text']

    def analyze_video(video_path, model, text_features):
        """
        Analyze a video and return the predicted class and confidence.

        Args:
            video_path (str): Path to the video file.
            model (sklearn.base.BaseEstimator): Pretrained model for classification.
            text_features (str): Additional text features.

        Returns:
            int: Predicted class.
            float: Confidence score of the prediction.
        """
        # Preprocess video
        video_frames = preprocess_video(video_path, N_FRAMES)
        video_frames = np.expand_dims(video_frames, axis=0)  # Add batch dimension

        # Flatten the video frames to use them as features (you might need to adjust this based on your model input)
        video_features = video_frames.flatten()

        # Tokenize text features
        encoded_text = tokenizer(text_features, padding='max_length', truncation=True, max_length=MAX_TEXT_FEATURES, return_tensors='pt')
        input_ids = encoded_text['input_ids'].flatten().numpy()  # Flatten and convert to numpy
        attention_mask = encoded_text['attention_mask'].flatten().numpy()

        # Combine video and text features
        combined_features = np.concatenate([video_features, input_ids, attention_mask])

        # Make predictions using the loaded model
        prediction = model.predict([combined_features])[0]
        confidence = max(model.predict_proba([combined_features])[0])

        return prediction, confidence

    # Transcribe the uploaded video
    text_features = transcribe_video(temp_file_path)

    # Analyze the uploaded video
    predicted_class, confidence = analyze_video(temp_file_path, model, text_features)

    # Display the results
    st.write(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
    st.write(f"Transcription: {text_features}")

else:
    st.write("No file uploaded yet.")