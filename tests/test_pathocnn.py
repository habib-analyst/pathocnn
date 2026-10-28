"""Tests for the PathoCNN release package.

Everything asserted here was measured against the shipped artifacts in
this environment; the metrics test compares the two JSON records
(training-time vs independently reproduced) instead of re-running the
full evaluation.
"""
import json
import os

import numpy as np
import pytest
import torch

from pathocnn import load_model, predict, predict_proba, preprocess
from pathocnn.infer import TRAIN_MEAN, TRAIN_STD
from pathocnn.model import build_model

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PKG, "assets")


@pytest.fixture(scope="module")
def data():
    return np.load(os.path.join(ASSETS, "pathmnist_test.npz"))


@pytest.fixture(scope="module")
def model_pt():
    return load_model(os.path.join(ASSETS, "pathocnn.pt"))


@pytest.fixture(scope="module")
def model_st():
    return load_model(os.path.join(PKG, "model.safetensors"))


def _batch(data, n=8):
    x = data["test_images"][:n].astype(np.float32) / 255.0
    mean = np.asarray(TRAIN_MEAN, dtype=np.float32)
    std = np.asarray(TRAIN_STD, dtype=np.float32)
    x = ((x - mean) / std).transpose(0, 3, 1, 2)
    return torch.from_numpy(x.astype(np.float32))


def test_param_count():
    # 288,809 trainable parameters; the state dict holds 289,454 values
    # once BatchNorm running statistics (buffers) are included.
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 288809
    assert sum(v.numel() for v in model.state_dict().values()) == 289454


def test_preprocess_shape_and_finiteness(data):
    t = preprocess(data["test_images"][0])
    assert t.shape == (1, 3, 28, 28)
    assert torch.isfinite(t).all()


def test_safetensors_matches_pytorch_checkpoint(model_pt, model_st, data):
    x = _batch(data)
    with torch.no_grad():
        a = model_pt(x)
        b = model_st(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_onnx_parity(model_pt, data):
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(os.path.join(PKG, "pathocnn.onnx"),
                                providers=["CPUExecutionProvider"])
    x = _batch(data, n=32).numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        ref = model_pt(torch.from_numpy(x)).numpy()
    assert np.abs(out - ref).max() < 1e-4


def test_demo_samples_measured_behavior(model_st):
    # Measured on the shipped weights (examples/predict_demo.py output):
    # 7 of the 9 first-of-class test patches are classified correctly;
    # smooth muscle and cancer-associated stroma are both read as debris.
    # The misses are pinned here on purpose — they are the model's
    # documented weak spot, not a bug in the demo.
    import glob
    expected_miss = {5: "debris", 7: "debris"}
    for path in sorted(glob.glob(os.path.join(ASSETS, "demo_samples", "*.png"))):
        cls = int(os.path.basename(path).split("_")[0])
        from pathocnn import LABELS
        r = predict(model_st, path)
        if cls in expected_miss:
            assert r["label"] == expected_miss[cls]
        else:
            assert r["label"] == LABELS[cls]


def test_first_adenocarcinoma_test_image(model_st, data):
    y = data["test_labels"].ravel()
    idx = int(np.where(y == 8)[0][0])
    r = predict(model_st, data["test_images"][idx])
    assert r["label"] == "colorectal adenocarcinoma epithelium"


def test_metrics_files_agree():
    saved = json.load(open(os.path.join(ASSETS, "pathocnn_metrics.json")))["test"]
    repro = json.load(open(os.path.join(ASSETS, "evaluation_results.json")))
    for k in ("accuracy", "auc_macro_ovr", "confusion_matrix",
              "majority_baseline", "per_class_recall"):
        assert repro[k] == saved[k], k


def test_weakest_class_is_cancer_associated_stroma():
    # The reproduced record's weakest recall is part of the model card's
    # honesty contract; pin it so a silent weights swap cannot hide it.
    repro = json.load(open(os.path.join(ASSETS, "evaluation_results.json")))
    rec = repro["per_class_recall"]
    assert min(rec, key=rec.get) == "cancer-associated stroma"
    assert rec["cancer-associated stroma"] == 0.3848
