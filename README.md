# Facial Emotion Recognition

This project is a real-time facial emotion recognition system built using deep learning and computer vision. The system detects a face from a webcam feed and classifies the facial expression into different emotion categories.

## Project Overview

The goal of this project is to recognize human emotions from facial expressions using CNN-based models and transfer learning. The project uses the FER-2013 dataset for training and evaluation.

The emotion classes used in this project are:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Features

- Real-time webcam-based emotion detection
- Face detection using OpenCV
- Emotion classification using deep learning models
- Baseline CNN model
- Transfer learning models
- Model evaluation using accuracy, classification report, and confusion matrix
- Comparison of multiple models

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Torchvision

## Project Structure

```text
Facial-Emotion-Recognition/
│
├── src/
│   ├── app/
│   │   └── webcam_app.py
│   │
│   ├── data/
│   │   └── load_data.py
│   │
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   └── transfer_models.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_resnet.py
│   │   ├── train_resnet50.py
│   │   ├── train_mobilenet.py
│   │   └── train_efficientnet.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       ├── confusion_matrix.py
│       ├── confusion_all_models.py
│       ├── classification_report.py
│       ├── sample_predictions.py
│       └── test_resnet50.py
│
├── requirements.txt
├── README.md
└── .gitignore
