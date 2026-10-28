"""PathoCNN demo: predict one bundled sample per class (native 28x28 test
patches, in-distribution), then print the first misclassified test images
with their confidences — the failures are part of the demo, not edited out.

This is a mechanics demo on a 28x28 benchmark model, not a clinical tool.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

from pathocnn import LABELS, load_model, predict  # noqa: E402
from pathocnn.infer import TRAIN_MEAN, TRAIN_STD  # noqa: E402

model = load_model(os.path.join(PKG, "model.safetensors"))

print("== bundled demo samples: first PathMNIST test patch of each class (28x28) ==")
for path in sorted(glob.glob(os.path.join(PKG, "assets", "demo_samples", "*.png"))):
    cls = int(os.path.basename(path).split("_")[0])
    r = predict(model, path)
    mark = "OK  " if r["label"] == LABELS[cls] else "MISS"
    print(f"[{mark}] true={LABELS[cls]:38s} -> predicted {r['label']:38s} "
          f"(confidence {r['confidence']:.1%})")

print("\n== first misclassified test patches (from assets/pathmnist_test.npz) ==")
import torch  # noqa: E402

d = np.load(os.path.join(PKG, "assets", "pathmnist_test.npz"))
x_test = d["test_images"]
y_test = d["test_labels"].ravel()
mean = np.asarray(TRAIN_MEAN, dtype=np.float32)
std = np.asarray(TRAIN_STD, dtype=np.float32)
shown = 0
with torch.no_grad():
    for i in range(0, len(x_test), 512):
        xb = (x_test[i:i + 512].astype(np.float32) / 255.0 - mean) / std
        xt = torch.from_numpy(xb.transpose(0, 3, 1, 2))
        probs = torch.softmax(model(xt), 1).numpy()
        for j, p in enumerate(probs):
            pred = int(p.argmax())
            if pred != int(y_test[i + j]) and shown < 5:
                print(f"test[{i + j}] true={LABELS[int(y_test[i + j])]} -> predicted "
                      f"{LABELS[pred]} (confidence {p[pred]:.1%})")
                shown += 1
        if shown >= 5:
            break
