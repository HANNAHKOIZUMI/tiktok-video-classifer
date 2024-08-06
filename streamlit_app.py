import streamlit as st
import tempfile
import torch
import cv2
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    MAX_TEXT_FEATURES = 3

    # Define the path to the saved model
    model_path = 'hybrid_model.pth'

    # Load and initialize the model
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Initialize the tokenizer
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

    # Example text input
    text_features = "Example text feature"

    # Analyze the uploaded video
    predicted_class, confidence = analyze_video(temp_file_path, model, text_features)

    # Display the results
    st.write(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

else:
    st.write("No file uploaded yet.")
