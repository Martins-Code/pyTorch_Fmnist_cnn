# scripts/preprocess.py
from PIL import Image
from io import BytesIO
from pathlib import Path
import torch
from torchvision import transforms
from typing import Union
import os

# Deterministic inference transforms (must match training normalization)
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def _open_image(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """Open an image from a path, bytes, or PIL.Image."""
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(BytesIO(image_input))
    else:
        # assume path-like (str/Path)
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img = Image.open(str(p))
    return img

def preprocess_image(image_input: Union[str, Path, bytes, Image.Image]) -> torch.Tensor:
    """
    Preprocess an image for the FashionMNIST model.
    Accepts:
      - path (str or Path)
      - bytes (file.read())
      - PIL.Image

    Returns:
      torch.Tensor of shape [1, 1, 28, 28]
    """
    img = _open_image(image_input)
    # Ensure grayscale (L) to be explicit
    img = img.convert("L")
    tensor = _transform(img)        # shape [1, 28, 28]
    tensor = tensor.unsqueeze(0)    # shape [1, 1, 28, 28]
    return tensor
