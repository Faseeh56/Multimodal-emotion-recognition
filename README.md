# Multimodal Emotion Recognition System

## Overview

This project implements a **Multimodal Emotion Recognition System** using deep learning and machine learning techniques. The system analyzes three different types of inputs — **text**, **images**, and **speech** — to predict the emotional state of the user. The system leverages models trained on text, facial expression images, and speech audio to provide a comprehensive emotion prediction.

## Features

* **Text Emotion Recognition**: Uses NLP techniques to classify emotions in text input.
* **Image Emotion Recognition**: Uses CNNs to analyze facial expressions and classify emotions.
* **Speech Emotion Recognition**: Uses MFCC features and machine learning models (Random Forest) to predict emotions from speech.
* **Multimodal Fusion**: Combines the outputs of all three modalities to provide a final, unified emotion prediction using **majority voting**.

## Technologies Used

* **Python** (Programming Language)
* **Streamlit** (Web Framework for UI)
* **TensorFlow/Keras** (Deep Learning Framework for Image Model)
* **scikit-learn** (Machine Learning Framework for Text and Speech Models)
* **Librosa** (Audio Processing Library for MFCC extraction)
* **joblib** (For loading machine learning models)
* **Pillow** (For image handling)

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

* Python 3.6+
* pip (Python package manager)

### Install Dependencies

To install the required Python packages, use the following commands:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the following libraries:

```bash
pip install streamlit tensorflow scikit-learn librosa pillow joblib
```

### Optional: Creating `requirements.txt`

You can create a `requirements.txt` file by running:

```bash
pip freeze > requirements.txt
```

## Project Structure

```bash
Multimodal_Emotion_Detection/
│
├── models/
│   ├── 02mobilenetv2_emotion_base.h5             # Image emotion model 
│   ├── class_labels.json                         # Class labels for image emotion model
│   ├── speech_emotion_rf.pkl                     # Speech emotion model 
│   ├── speech_label_encoder.pkl                  # Label encoder for speech model
│   ├── text_emotion_model.pkl                    # Text emotion model 
│   ├── tfidf_vectorizer.pkl                      # TF-IDF vectorizer for text
│   └── text_label_encoder.pkl                    # Label encoder for text model
│
├── app.py                                        # Main Streamlit app
├── README.md                                     # Project overview and documentation
└── requirements.txt                              # List of dependencies
```

## How to Run the Application

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Faseeh56/Multimodal_Emotion_Detection.git
   cd Multimodal_Emotion_Detection
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. **Open the app** in your browser at `http://localhost:8501`.

## Usage

Once the app is running, you can interact with it by providing **text**, **image**, or **speech** inputs for emotion analysis.

* **Text Input**: Enter a piece of text in the provided text box, and the app will predict the emotion conveyed by the text.
* **Image Input**: Upload an image of a face, and the app will analyze the facial expression to predict the emotion.
* **Speech Input**: Upload an audio file in `.wav` or `.mp3` format, and the app will analyze the speech to detect emotions based on the tone of voice.

After submitting the inputs, the app will display individual predictions for text, image, and speech, as well as a **final emotion prediction** based on **majority voting**.

## Fusion Method

The final emotion prediction is determined by **majority voting** across the text, image, and speech predictions:

* The emotion that appears most frequently across the three modalities is chosen as the final prediction.

## Future Improvements

* **Real-time Emotion Recognition**: Integrate a real-time webcam or microphone input for live emotion detection.
* **Better Fusion Strategies**: Implement advanced fusion strategies, like weighted voting or neural network-based fusion.
* **Model Optimizations**: Fine-tune the models to improve performance.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests if you have improvements or fixes. Contributions are welcome!


