import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from utils import predict_emotion

# Emotion → Emoji Map
EMOJI_MAP = {
    "Angry": "😡",
    "Happy": "😄",
    "Neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
}

st.set_page_config(page_title="Live Emotion Detection", page_icon="😊")

st.markdown("<h1 style='text-align:center;'>😊 Live Facial Emotion Recognition</h1>", unsafe_allow_html=True)
st.write("Turn on your webcam and let the model detect your emotion!")

class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                                  "haarcascade_frontalface_default.xml")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no faces found, return original frame
        if len(faces) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]

            # Save the cropped face temporarily
            cv2.imwrite("live_face.jpg", face_img)

            # Predict Emotion
            emotion, confidence, scores = predict_emotion("live_face.jpg")

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            # Display result text
            label = f"{EMOJI_MAP[emotion]} {emotion} ({confidence:.2f})"
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------------
# Start Webcam Streamlit WebRTC
# -----------------------------------
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Powered by MobileNetV2 – Fine-Tuned for Facial Expressions</p>", unsafe_allow_html=True)
