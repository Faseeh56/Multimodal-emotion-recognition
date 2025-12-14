import streamlit as st
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json
import librosa
import librosa.display

# Load models
text_model = joblib.load(r"F:\Multimodal_Emotion_Detection\models\text_emotion_model.pkl")
text_tfidf = joblib.load(r"F:\Multimodal_Emotion_Detection\models\tfidf_vectorizer.pkl")
text_le = joblib.load(r"F:\Multimodal_Emotion_Detection\models\text_label_encoder.pkl")
image_model = load_model(r"F:\Multimodal_Emotion_Detection\models\02mobilenetv2_emotion_base.h5")

with open(r"F:\Multimodal_Emotion_Detection\models\class_labels.json", "r") as f:
    image_classes = json.load(f)

speech_model = joblib.load(r"F:\Multimodal_Emotion_Detection\models\speech_emotion_rf.pkl")
speech_le = joblib.load(r"F:\Multimodal_Emotion_Detection\models\speech_label_encoder.pkl")

CANONICAL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def align_proba(proba, model_classes):
    model_classes = [c.lower() for c in model_classes]
    idx = {c: i for i, c in enumerate(model_classes)}
    aligned = np.zeros(len(CANONICAL))
    for i, c in enumerate(CANONICAL):
        aligned[i] = proba[idx[c]]
    return aligned

def predict_text(text):
    X = text_tfidf.transform([text])
    proba = text_model.predict_proba(X)[0]
    proba = align_proba(proba, text_le.classes_)
    label = CANONICAL[np.argmax(proba)]
    return label, np.max(proba), proba

def predict_image(img_file):
    img = Image.open(img_file).convert("L").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.stack([arr, arr, arr], axis=-1) / 255.0
    x = np.expand_dims(arr, axis=0)

    proba = image_model.predict(x)[0]
    proba = align_proba(proba, image_classes)
    label = CANONICAL[np.argmax(proba)]
    return label, np.max(proba), proba

def extract_mfcc_features(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

def predict_speech(audio_file):
    mfcc_features = extract_mfcc_features(audio_file)
    mfcc_features = mfcc_features.reshape(1, -1)
    proba = speech_model.predict_proba(mfcc_features)[0]
    predicted_class = speech_le.inverse_transform([np.argmax(proba)])[0]
    proba = align_proba(proba, CANONICAL)
    label = CANONICAL[np.argmax(proba)]
    conf = np.max(proba)
    return label, conf, proba

# ---------------- UI ----------------
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Multimodal Emotion Recognition")

# UI containers for different sections
text = st.text_area("Enter text for emotion analysis")
img_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])
audio_file = st.file_uploader("Upload speech audio", type=["wav", "mp3"])

# Button to trigger all predictions
if st.button("Predict Emotions for All"):
    predictions = {"text": None, "image": None, "speech": None}

    if not text.strip() and img_file is None and audio_file is None:
        st.error("Please enter text or upload an image/audio file for prediction.")
    else:
        # Displaying the Text Emotion Prediction
        if text.strip():
            label, conf, proba = predict_text(text)
            predictions["text"] = (label, conf, proba)
            with st.expander("Text Emotion Prediction"):
                st.success(f"Emotion: **{label}** ({conf:.2f})")
                st.bar_chart(proba)
        else:
            st.warning("No text input provided.")

        # Displaying the Image Emotion Prediction
        if img_file is not None:
            label, conf, proba = predict_image(img_file)
            predictions["image"] = (label, conf, proba)
            with st.expander("Image Emotion Prediction"):
                st.success(f"Emotion: **{label}** ({conf:.2f})")
                st.bar_chart(proba)
        else:
            st.warning("No image uploaded.")

        # Displaying the Speech Emotion Prediction
        if audio_file is not None:
            label, conf, proba = predict_speech(audio_file)
            predictions["speech"] = (label, conf, proba)
            with st.expander("Speech Emotion Prediction"):
                st.success(f"Emotion: **{label}** ({conf:.2f})")
                st.bar_chart(proba)
        else:
            st.warning("No audio uploaded.")

        # Final Prediction (Majority Voting or Weighted)
        # Majority Voting: Count the most frequent label
        all_labels = [pred[0] for pred in predictions.values() if pred is not None]
        if all_labels:
            final_prediction = max(set(all_labels), key=all_labels.count)
            st.subheader(f"Final Predicted Emotion: **{final_prediction}**")
        else:
            st.warning("No valid predictions to combine.")
