---
license: mit
pipeline_tag: image-classification
language: en
tags:
  - medical-imaging
  - histology
  - colorectal
  - medmnist
  - pathmnist
  - pytorch
  - onnx
  - safetensors
---

# PathoCNN — PathMNIST colorectal histology classifier

**A compact CNN (288,809 parameters), designed and trained from scratch for 9-class classification of 28×28 H&E colorectal histology patches, released with something most model uploads don't have: a script that re-derives every headline metric from the shipped weights and data, so the numbers below are checkable, not just claimed.**

- Test accuracy **0.8712** and macro one-vs-rest AUC **0.9855** on the PathMNIST test split (7,180 patches), vs a 0.1864 majority-class baseline — independently reproduced from the artifacts in this repo on 2026-10-28 (see *Verify it yourself*).
- The test split is **CRC-VAL-HE-7K, collected at a different clinical center** than the training data — so these numbers include a real cross-center generalization gap (validation accuracy on same-center data was 0.9629). That gap is part of the result, not a footnote.
- Weights in three formats: `model.safetensors`, the training checkpoint (`assets/pathocnn.pt`), and ONNX (`pathocnn.onnx`, parity vs PyTorch measured at 5.7e-06 max logit difference).
- Trained from scratch by Habib Ur Rehman; sibling release to SonoCNN (BreastMNIST).

> **Research and benchmark use only. This is a 28×28 benchmark model. It is not a medical device, it has never been evaluated on clinical whole-slide data, and it must not be used for diagnosis or screening.**

## Verified results — PathMNIST test split (CRC-VAL-HE-7K, n = 7,180)

| Metric | Value | Source |
|---|---|---|
| Accuracy | **0.8712** (6,255/7,180) | reproduced 2026-10-28 |
| AUC (macro, one-vs-rest) | **0.9855** | reproduced 2026-10-28 |
| Balanced accuracy | 0.8251 | reproduced 2026-10-28 |
| Majority-class baseline | 0.1864 | reproduced 2026-10-28 |
| Best validation accuracy | 0.9629 (epoch 11) | recorded at training time |
| Best validation AUC | 0.9987 | recorded at training time |

Per-class recall on the test split (all reproduced):

| Class | Recall | | Class | Recall |
|---|---|---|---|---|
| adipose | 0.9477 | | smooth muscle | 0.7348 |
| background | 1.0000 | | normal colon mucosa | 0.8516 |
| debris | 0.7817 | | **cancer-associated stroma** | **0.3848** |
| lymphocytes | 0.9101 | | colorectal adenocarcinoma epithelium | 0.9667 |
| mucus | 0.8483 | | | |

Read the weak row honestly: **the model largely fails on cancer-associated stroma** — it recalls only 162 of 421 stroma patches, and reads 155 of them as *debris*. Debris is the model's error sink: debris precision is just 0.4417, meaning most "debris" predictions are wrong (they are mostly stroma and smooth muscle). The clinically most important class, adenocarcinoma epithelium, is also the best detected (recall 0.9667). Accuracy's 95% confidence interval on 7,180 patches is roughly ±0.8 points (≈0.863–0.879). The full 9×9 confusion matrix and per-class precision are in `assets/evaluation_results.json`; the training-time record is in `assets/pathocnn_metrics.json`; they agree exactly on every headline number.

Context, not a claim of parity: the MedMNIST v2 paper's reference ResNet-18 (11.7M parameters, ~40× larger) reports 0.907 accuracy / 0.983 AUC on this split, and ResNet-50 reports 0.911 / 0.990. PathoCNN's AUC sits in that band; its accuracy trails by ~3.6 points, which is the price of the small model and shows up almost entirely in the stroma/debris/muscle confusions above.

## Quickstart

```bash
pip install "git+https://github.com/habib-analyst/pathocnn.git"
```

```python
from pathocnn import load_model, predict

model = load_model("model.safetensors")   # or assets/pathocnn.pt
result = predict(model, "assets/demo_samples/8_colorectal_adenocarcinoma_epithelium.png")
# {'label': 'colorectal adenocarcinoma epithelium', 'confidence': 0.999,
#  'probabilities': {...9 classes...}}
```

Preprocessing contract (identical to training): RGB → resize to 28×28 if needed → /255 → normalize per channel with the PathMNIST train-split mean/std (**[0.74055, 0.53298, 0.70583] / [0.12368, 0.17676, 0.12443]**). Labels are the MedMNIST v2 PathMNIST order 0–8 (see `config.json`).

ONNX: `pathocnn.onnx` takes the same normalized `(N, 3, 28, 28)` float tensor and returns logits.

## Verify it yourself

```bash
python examples/verify_metrics.py
```

This re-evaluates the shipped weights on the bundled test split (`assets/pathmnist_test.npz`) and compares against the training-time metrics file. On 2026-10-28 it printed `ALL HEADLINE METRICS REPRODUCED` — accuracy, AUC, the full confusion matrix, and every per-class recall matched exactly. One protocol warning the script and training code both encode: the train-split normalization statistics must be computed with chunked float64 accumulation. A single float32 NumPy reduction over ~90k images silently returns wrong channel means on this dataset (measured: 0.23778 for every channel instead of 0.74055/0.53298/0.70583) — the kind of silent error that makes "reproductions" fail for reasons nobody can see.

```bash
python examples/predict_demo.py   # one patch per class + first misclassifications
python -m pytest -q               # 8 tests: formats agree, metrics files agree, parity
```

The demo is honest about the weak spot: of the nine first-of-class test patches, seven are classified correctly, and the two misses — smooth muscle and cancer-associated stroma — are both confidently read as debris (95.4% and 99.7%). The demo prints them, and the test suite pins them.

## Model details

- **Architecture:** 3 conv blocks (32→64→128 channels, BatchNorm, ReLU, max-pool, dropout) + 2-layer MLP head, designed for this release in the same compact family as SonoCNN. 288,809 trainable parameters; 1.2 MB in safetensors.
- **Training:** from scratch on the PathMNIST train split (89,996 patches), 12 epochs, AdamW (lr 2e-3, weight decay 1e-4, OneCycle cosine schedule), batch 256, plain cross-entropy (the train split's max/min class ratio is 1.63, so no class weighting), seed 42, checkpoint selected by best validation accuracy (epoch 11 of 12). Training took 1,990 s (~33 min) on a 2-core CPU. The exact script is `examples/train.py`.
- **Augmentation:** random dihedral transforms (90° rotations and flips) on the training split only. Histology patches have no canonical orientation, so the full D4 group is label-preserving here. No color jitter — H&E stain color carries class signal and was left alone.

## What PathoCNN is not

- **Not a clinical tool.** It classifies 28×28 benchmark patches, not whole slides or patients, and its weakest class (cancer-associated stroma, 38.5% recall) is a tissue type a real workflow cares about. The cross-center drop from 0.9629 (validation) to 0.8712 (test) is what domain shift does to small models; expect worse on any other center.
- **Not state of the art.** Larger reference models score ~0.91 accuracy on this split. This release exists because its numbers are verified and its failure modes are documented, not because it tops a leaderboard.
- **Not trained on private or clinical data.** Only the public PathMNIST benchmark (below).
- **Confidences are not calibrated.** Softmax outputs are frequently extreme (several demo patches score a rounded 100.0%), including on the confidently-wrong stroma→debris errors. Treat them as rankings, not probabilities of being right.

## Data and attribution

PathMNIST is part of **MedMNIST v2** (Yang et al., *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 2023). It derives from the NCT-CRC-HE-100K dataset of 100,000 H&E histology patches (Kather et al., 2019), split 9:1 into train/validation, with CRC-VAL-HE-7K as the external-center test set. The dataset is licensed **CC BY 4.0**; `assets/pathmnist_test.npz` (the official test split only) is redistributed here for metric reproduction, with attribution. The source file used for training is Zenodo record 10519652, md5 `a8b06965200029087d5bd730944a56c1`. The MIT license of this repository covers the code and the trained weights, not the dataset.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Release weights (safetensors) |
| `pathocnn.onnx` | ONNX export (logits) |
| `config.json` | Architecture + preprocessing config |
| `src/pathocnn/` | `model.py` (architecture), `infer.py` (loading + prediction API) |
| `examples/train.py` | The exact training script that produced the weights |
| `examples/predict_demo.py` | Per-class demo + first misclassifications, printed honestly |
| `examples/verify_metrics.py` | Full metric reproduction from the artifacts |
| `assets/pathocnn.pt` | Training checkpoint (state dict + training record) |
| `assets/pathocnn_metrics.json` | Training-time metrics record |
| `assets/evaluation_results.json` | Independently reproduced metrics (2026-10-28) |
| `assets/pathmnist_test.npz` | PathMNIST official test split (CC BY 4.0, see attribution) |
| `assets/demo_samples/` | First test patch of each class (28×28 PNGs) |

## License

MIT for code and weights (see `LICENSE`). Dataset under CC BY 4.0 as above.
