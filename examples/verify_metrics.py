"""Verify every headline metric of the PathoCNN release from the artifacts
themselves: re-evaluate the shipped weights (model.safetensors) on the
bundled PathMNIST test split and compare against the metrics recorded at
training time (assets/pathocnn_metrics.json).

Protocol (identical to training-time evaluation in examples/train.py):
  images / 255 -> normalize per channel with the train-split mean/std
  (pathocnn.infer.TRAIN_MEAN / TRAIN_STD) -> PathoCNN forward, batch 512,
  no augmentation. The test split is CRC-VAL-HE-7K: patches from a
  different clinical center than the training data.

Usage:
  python examples/verify_metrics.py [--weights model.safetensors]
      [--data assets/pathmnist_test.npz]
      [--saved-metrics assets/pathocnn_metrics.json]
      [--out assets/evaluation_results.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from pathocnn.infer import LABELS, TRAIN_MEAN, TRAIN_STD  # noqa: E402
from pathocnn.model import build_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
NUM_CLASSES = 9


@torch.no_grad()
def evaluate(model, images_u8, batch=512):
    model.eval()
    xt = torch.from_numpy(images_u8).permute(0, 3, 1, 2)
    mean = torch.tensor(TRAIN_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(TRAIN_STD).view(1, 3, 1, 1)
    probs = []
    for i in range(0, len(xt), batch):
        xb = (xt[i:i + batch].float() / 255.0 - mean) / std
        probs.append(torch.softmax(model(xb), dim=1))
    probs = torch.cat(probs).numpy()
    return probs, probs.argmax(1)


def full_metrics(y_true, probs, preds):
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))
    rec = recall_score(y_true, preds, labels=list(range(NUM_CLASSES)),
                       average=None, zero_division=0)
    prec = precision_score(y_true, preds, labels=list(range(NUM_CLASSES)),
                           average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "auc_macro_ovr": round(float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": {LABELS[i]: round(float(rec[i]), 4) for i in range(NUM_CLASSES)},
        "per_class_precision": {LABELS[i]: round(float(prec[i]), 4) for i in range(NUM_CLASSES)},
        "class_counts": np.bincount(y_true, minlength=NUM_CLASSES).tolist(),
        "majority_baseline": round(float(np.bincount(y_true).max() / len(y_true)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(PKG, "model.safetensors"))
    ap.add_argument("--data", default=os.path.join(PKG, "assets", "pathmnist_test.npz"))
    ap.add_argument("--saved-metrics", default=os.path.join(PKG, "assets", "pathocnn_metrics.json"))
    ap.add_argument("--out", default=None, help="optional path to write the reproduced metrics JSON")
    args = ap.parse_args()

    d = np.load(args.data)
    x_test = d["test_images"]
    y_test = d["test_labels"].ravel().astype(np.int64)

    from safetensors.torch import load_file
    model = build_model()
    model.load_state_dict(load_file(args.weights))
    model.eval()

    probs, preds = evaluate(model, x_test)
    res = {"n_test": int(len(y_test)), **full_metrics(y_test, probs, preds)}
    print(json.dumps(res, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out}")

    if os.path.exists(args.saved_metrics):
        saved = json.load(open(args.saved_metrics))["test"]
        print("\ncomparison vs training-time metrics file:")
        ok = True
        for k in ("accuracy", "auc_macro_ovr", "confusion_matrix",
                  "majority_baseline", "per_class_recall"):
            match = res[k] == saved[k]
            ok &= match
            print(f"  {k}: match={match}  reproduced={res[k] if k != 'confusion_matrix' else '<9x9 matrix>'}")
        print("ALL HEADLINE METRICS REPRODUCED" if ok else "MISMATCH — see above")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
