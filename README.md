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

## How to Run This Project

Follow these steps to run the project after downloading or cloning the repository.

### 1. Clone the Repository

```bash
git clone https://github.com/ChaitraKrothapalli/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

### 2. Create a Virtual Environment

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the FER-2013 dataset from Kaggle and place it inside a local `data/` folder.

The dataset is not included in this GitHub repository because it is large.

Expected local structure:

```text
data/
├── train/
├── val/
└── test/
```

or use the same dataset structure required by `src/data/load_data.py`.

### 5. Train the Models

Run these commands from the main project folder.

To train the baseline CNN model:

```bash
python src/training/train_baseline.py
```

To train the transfer learning models:

```bash
python src/training/train_resnet.py
python src/training/train_resnet50.py
python src/training/train_mobilenet.py
python src/training/train_efficientnet.py
```

### 6. Evaluate the Models

To evaluate a trained model:

```bash
python src/evaluation/evaluate.py
```

To generate a confusion matrix:

```bash
python src/evaluation/confusion_matrix.py
```

To generate confusion matrices for all models:

```bash
python src/evaluation/confusion_all_models.py
```

To generate a classification report:

```bash
python src/evaluation/classification_report.py
```

To test the ResNet50 model:

```bash
python src/evaluation/test_resnet50.py
```

### 7. Run the Real-Time Webcam Demo

After training the model or placing the trained model checkpoint inside the local `models/` folder, run:

```bash
python src/app/webcam_app.py
```

The webcam application will open the camera, detect the face, and display the predicted emotion in real time.

Press `q` to close the webcam window. On Mac, click the webcam window first and then press `q`. If it still does not close, stop the program from the terminal using:

```bash
Ctrl + C
```

## Important Note

The following folders are not included in this GitHub repository:

```text
data/
models/
outputs/
venv/
```

These folders are ignored because they contain large files, generated outputs, or local environment files.

To fully run the project, the user must download the dataset locally and either train the models again or place trained model checkpoint files inside the local `models/` folder.
