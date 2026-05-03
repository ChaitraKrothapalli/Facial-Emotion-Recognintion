import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)

if torch.backends.mps.is_available():
    print("Device: Apple MPS available")
elif torch.cuda.is_available():
    print("Device: CUDA available")
else:
    print("Device: CPU")

print("Setup working ")