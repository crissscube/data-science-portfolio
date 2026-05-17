import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.conv3(self.conv2(self.conv1(x))))


# MNIST normalization constants
MEAN, STD = 0.1307, 0.3081

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
])


def load_model(checkpoint_path: str = 'digit_cnn_checkpoint.pth') -> nn.Module:
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(model: nn.Module, image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # If the image has a white background, invert it to match MNIST (black bg, white digit)
    grayscale = ImageOps.grayscale(image)
    if np.array(grayscale).mean() > 127:
        image = ImageOps.invert(image)

    tensor = transform(image).unsqueeze(0)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze()

    predicted_digit = int(probs.argmax().item())
    confidence = float(probs[predicted_digit].item())
    all_probs = {str(i): round(float(probs[i].item()), 4) for i in range(10)}

    return {
        'digit': predicted_digit,
        'confidence': round(confidence, 4),
        'probabilities': all_probs
    }
