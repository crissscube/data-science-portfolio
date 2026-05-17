"""
Test client for the Digit Recognition API.
Pulls 10 random images from MNIST and sends them to the running API.

Usage:
    python client_test.py
"""

import requests
import io
import numpy as np
from PIL import Image

API_URL = 'http://localhost:8000'


def mnist_image_to_bytes(pixel_array: np.ndarray) -> bytes:
    """Convert a 28x28 numpy array (0-1 float) to PNG bytes."""
    img = Image.fromarray((pixel_array * 255).astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_health():
    resp = requests.get(f'{API_URL}/health')
    print(f'Health check: {resp.json()}')


def test_predict_mnist():
    import torchvision
    import torchvision.transforms as transforms

    dataset = torchvision.datasets.MNIST(
        '../data', train=False, download=True, transform=transforms.ToTensor()
    )

    print(f'\n{"Idx":>5} | {"True":>5} | {"Pred":>5} | {"Confidence":>11} | {"Match":>5}')
    print('-' * 45)

    correct = 0
    indices = np.random.choice(len(dataset), size=10, replace=False)

    for idx in indices:
        img_tensor, true_label = dataset[idx]
        img_array = img_tensor.squeeze().numpy()  # (28, 28)
        img_bytes = mnist_image_to_bytes(img_array)

        resp = requests.post(
            f'{API_URL}/predict',
            files={'file': ('digit.png', img_bytes, 'image/png')}
        )
        result = resp.json()
        pred = result['predicted_digit']
        conf = result['confidence']
        match = '✓' if pred == true_label else '✗'
        if pred == true_label:
            correct += 1

        print(f'{idx:>5} | {true_label:>5} | {pred:>5} | {conf:>11} | {match:>5}')

    print(f'\nAccuracy on sample: {correct}/10')


if __name__ == '__main__':
    test_health()
    test_predict_mnist()
