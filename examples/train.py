"""Train PathoCNN from scratch on PathMNIST — the exact script that
produced the released weights.

Protocol:
  - PathMNIST official splits (train 89,996 / val 10,004 / test 7,180),
    downloaded from Zenodo (record 10519652, md5 a8b06965200029087d5bd730944a56c1).
  - Inputs: /255, normalized per channel with the train-split mean/std
    (the rounded contract values in pathocnn.infer.TRAIN_MEAN/TRAIN_STD).
  - Augmentation (train only): random dihedral transforms — 90-degree
    rotations and flips. Histology patches have no canonical
    orientation, so the full D4 group is label-preserving for this
    data. No color jitter: H&E stain color carries class signal.
  - AdamW (lr 2e-3, weight decay 1e-4, cosine schedule), batch 256,
    12 epochs, plain cross-entropy (train split max/min class ratio
    is 1.63, so no class weighting), seed 42 throughout.
  - Checkpoint selected by best validation accuracy; the test split is
    evaluated exactly once, after selection, and recorded in the
    metrics JSON. examples/verify_metrics.py later reproduces those
    test numbers from the shipped weights.

Usage:
  python examples/train.py --data /path/to/pathmnist.npz --out /path/to/workdir

If the npz is missing, download it first, e.g.:
  curl -L -o pathmnist.npz "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
(Python HTTP clients can stall on Zenodo through some proxies; curl is reliable.)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from pathocnn.infer import LABELS, TRAIN_MEAN, TRAIN_STD  # noqa: E402
from pathocnn.model import build_model  # noqa: E402

NUM_CLASSES = 9


def compute_split_stats(images_u8):
    """Per-channel mean/std with chunked float64 accumulation.

    A single float32 numpy reduction over ~90k images silently returns
    wrong statistics at this size (measured on this dataset: channel
    means of 0.23778/0.23778/0.23778 instead of 0.74055/0.53298/0.70583).
    Chunked float64 is the safe route; see REPORT.md.
    """
    n = 0
    s = np.zeros(3)
    ss = np.zeros(3)
    for i in range(0, len(images_u8), 10000):
        xb = images_u8[i:i + 10000].astype(np.float64) / 255.0
        n += xb.shape[0] * 28 * 28
        s += xb.sum(axis=(0, 1, 2))
        ss += (xb ** 2).sum(axis=(0, 1, 2))
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    return mean, std


def normalize(x_u8_nchw):
    """uint8 NCHW tensor -> normalized float tensor (contract stats)."""
    x = x_u8_nchw.float() / 255.0
    mean = torch.tensor(TRAIN_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(TRAIN_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x_nchw, gen):
    """Random dihedral transform per sample (rot90 k in 0..3 + optional h-flip)."""
    n = x_nchw.shape[0]
    ks = torch.randint(0, 4, (n,), generator=gen)
    flips = torch.randint(0, 2, (n,), generator=gen)
    out = torch.empty_like(x_nchw)
    for k in range(4):
        for f in range(2):
            idx = (ks == k) & (flips == f)
            if idx.any():
                t = torch.rot90(x_nchw[idx], k, dims=(2, 3))
                if f:
                    t = torch.flip(t, dims=(3,))
                out[idx] = t
    return out


@torch.no_grad()
def evaluate(model, images_u8, labels, batch=512):
    """Return (probs np array, preds np array) for a split, contract preprocessing."""
    model.eval()
    xt = torch.from_numpy(images_u8).permute(0, 3, 1, 2)
    probs = []
    for i in range(0, len(xt), batch):
        xb = normalize(xt[i:i + batch])
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
        "n": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to pathmnist.npz")
    ap.add_argument("--out", required=True, help="work dir for checkpoint + metrics")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    d = np.load(args.data)
    x_train_u8 = d["train_images"]
    y_train = d["train_labels"].ravel().astype(np.int64)
    x_val_u8 = d["val_images"]
    y_val = d["val_labels"].ravel().astype(np.int64)
    x_test_u8 = d["test_images"]
    y_test = d["test_labels"].ravel().astype(np.int64)

    exact_mean, exact_std = compute_split_stats(x_train_u8)
    print(f"train stats (chunked float64): mean={exact_mean.round(5).tolist()} "
          f"std={exact_std.round(5).tolist()}", flush=True)
    print(f"contract stats used: mean={list(TRAIN_MEAN)} std={list(TRAIN_STD)}", flush=True)
    assert np.abs(exact_mean - np.array(TRAIN_MEAN)).max() < 1e-4
    assert np.abs(exact_std - np.array(TRAIN_STD)).max() < 1e-4

    xt = torch.from_numpy(x_train_u8).permute(0, 3, 1, 2)  # uint8 NCHW in RAM
    yt = torch.from_numpy(y_train)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(xt) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
        pct_start=0.1, anneal_strategy="cos")
    lossf = nn.CrossEntropyLoss()

    log = []
    best_val_acc = -1.0
    best_path = os.path.join(args.out, "pathocnn.pt")
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        t0 = time.time()
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(normalize(xt[idx]), gen)
            yb = yt[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item() * len(idx)
            ep_correct += (out.argmax(1) == yb).sum().item()
            ep_n += len(idx)
        val_probs, val_preds = evaluate(model, x_val_u8, y_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_auc = float(roc_auc_score(y_val, val_probs, multi_class="ovr", average="macro"))
        row = {
            "epoch": epoch + 1,
            "train_loss": round(ep_loss / ep_n, 4),
            "train_acc": round(ep_correct / ep_n, 4),
            "val_acc": round(val_acc, 4),
            "val_auc_macro_ovr": round(val_auc, 4),
            "epoch_seconds": round(time.time() - t0, 1),
        }
        log.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_auc_macro_ovr": val_auc,
                "seed": args.seed,
                "train_mean": list(TRAIN_MEAN),
                "train_std": list(TRAIN_STD),
            }, best_path)
            print(f"  saved best checkpoint (val_acc {val_acc:.4f})", flush=True)

    total_seconds = round(time.time() - t_start, 1)

    # Final: load the best-val checkpoint and evaluate the test split once.
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_probs, test_preds = evaluate(model, x_test_u8, y_test)
    metrics = {
        "model": "PathoCNN",
        "version": "0.1.0",
        "dataset": "PathMNIST (MedMNIST v2), official splits",
        "seed": args.seed,
        "epochs_run": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "best_epoch": ckpt["epoch"],
        "best_val_acc": round(float(ckpt["val_acc"]), 4),
        "best_val_auc_macro_ovr": round(float(ckpt["val_auc_macro_ovr"]), 4),
        "train_seconds_total": total_seconds,
        "train_mean_exact": exact_mean.round(5).tolist(),
        "train_std_exact": exact_std.round(5).tolist(),
        "test": full_metrics(y_test, test_probs, test_preds),
        "epoch_log": log,
    }
    with open(os.path.join(args.out, "pathocnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2), flush=True)
    print(f"DONE total_seconds={total_seconds} best_epoch={ckpt['epoch']} "
          f"best_val_acc={best_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
