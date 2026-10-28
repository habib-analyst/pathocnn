"""Inference helpers for PathoCNN.

Preprocessing contract (identical to training/evaluation):
  RGB -> resize to 28x28 (box interpolation) if needed -> /255
  -> normalize per channel with the PathMNIST train-split mean/std
     (TRAIN_MEAN / TRAIN_STD below, computed on the full train split
     with chunked float64 accumulation and re-verified by
     examples/verify_metrics.py).

Labels: MedMNIST v2 PathMNIST order, see pathocnn.model.
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
from PIL import Image

from .model import PathoCNN, build_model

# Full-train-split statistics (89,996 images), computed with chunked
# float64 accumulation. NOTE: a naive float32 numpy reduction over the
# whole split silently returns wrong values on this dataset size
# (measured: mean 0.23778 for every channel instead of the values
# below) -- see REPORT.md. Do not "recompute" these in float32.
TRAIN_MEAN = (0.74055, 0.53298, 0.70583)
TRAIN_STD = (0.12368, 0.17676, 0.12443)
LABELS = (
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
)
IMG_SIZE = 28

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def load_model(weights_path: str | os.PathLike, device: str = "cpu") -> PathoCNN:
    """Load a PathoCNN checkpoint (.safetensors or release .pt) in eval mode."""
    model = build_model()
    path = str(weights_path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(image: ImageLike) -> torch.Tensor:
    """One image -> (1, 3, 28, 28) normalized float tensor."""
    if isinstance(image, (str, os.PathLike)):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    img = img.convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    x = np.asarray(img).astype(np.float32) / 255.0
    mean = np.asarray(TRAIN_MEAN, dtype=np.float32)[None, None, :]
    std = np.asarray(TRAIN_STD, dtype=np.float32)[None, None, :]
    x = ((x - mean) / std).astype(np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def predict_proba(model: PathoCNN, image: ImageLike) -> dict:
    """Return {label: probability} for all 9 classes."""
    x = preprocess(image)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    return {LABELS[i]: probs[i] for i in range(len(LABELS))}


def predict(model: PathoCNN, image: ImageLike) -> dict:
    """Return {'label', 'confidence', 'probabilities'} for one image."""
    probs = predict_proba(model, image)
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probabilities": probs}
