# scripts/predict.py
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Make project root importable no matter where the script is run from
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now safe to import local modules
from scripts.preprocess import preprocess_image
from scripts.model import FashionClassifier
from typing import Union


# class labels (FashionMNIST)
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_model(weights_path: Union[str, Path] = None, device: torch.device = None):
    """
    Load the model and return (model, device).
    - weights_path: if None, uses ROOT/models/fashion_mnist.pth
    - device: if None, uses cuda if available else cpu
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_path is None:
        weights_path = ROOT / "models" / "fashion_mnist.pth"
    weights_path = Path(weights_path)

    model = FashionClassifier()
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device

# We'll load model once at import time for convenience (but you can call load_model yourself)
_model, _device = load_model()

def _predict_tensor(image_tensor: torch.Tensor, model=_model, device=_device):
    """
    image_tensor: [1,1,28,28] (cpu or device)
    returns: (index, label, confidence)
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return int(idx.item()), CLASSES[int(idx.item())], float(conf.item())

def predict_from_path(image_path: str, model=_model, device=_device):
    t = preprocess_image(image_path)   # accepts path
    return _predict_tensor(t, model=model, device=device)

def predict_from_bytes(image_bytes: bytes, model=_model, device=_device):
    t = preprocess_image(image_bytes)  # accepts bytes
    return _predict_tensor(t, model=model, device=device)

# CLI helper (keeps predict API clean for programmatic usage)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict FashionMNIST class from image")
    parser.add_argument("image", type=str, help="Path to image file")
    args = parser.parse_args()

    idx, label, conf = predict_from_path(args.image)
    print(f"Predicted index: {idx}")
    print(f"Predicted label: {label}")
    print(f"Confidence: {conf:.3f}")
