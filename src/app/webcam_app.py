import sys
import os
sys.path.append(os.path.abspath("."))

from collections import deque, Counter

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.models.transfer_models import get_resnet50


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class_names = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]

emotion_colors = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (255, 165, 0),
    "neutral": (200, 200, 200)
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = get_resnet50(num_classes=7)
model.load_state_dict(torch.load("models/best_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    sys.exit()

prediction_history = deque(maxlen=10)

window_name = "Real-Time Facial Emotion Recognition"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("Webcam started.")
print("Click on the webcam window and press q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_image = Image.fromarray(face)

            input_tensor = transform(face_image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)

            current_emotion = class_names[predicted.item()]
            prediction_history.append(current_emotion)

            emotion = Counter(prediction_history).most_common(1)[0][0]
            color = emotion_colors[emotion]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")