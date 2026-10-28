"""PathoCNN — compact CNN for PathMNIST colorectal histology classification.

A research/benchmark model release (not a medical device). See the model
card (README.md) for verified metrics and limitations.
"""
from .infer import LABELS, TRAIN_MEAN, TRAIN_STD, load_model, predict, predict_proba, preprocess
from .model import PathoCNN, build_model

__version__ = "0.1.0"
__all__ = [
    "PathoCNN",
    "build_model",
    "load_model",
    "predict",
    "predict_proba",
    "preprocess",
    "LABELS",
    "TRAIN_MEAN",
    "TRAIN_STD",
]
