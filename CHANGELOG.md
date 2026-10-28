# Changelog

## 0.1.0 - 2026-10-28

- Initial release of PathoCNN as a standalone model package.
- From-scratch PathMNIST model (288,809 parameters) trained for this
  release: 12 epochs on the official 89,996-patch train split, seed 42,
  dihedral augmentation, checkpoint selected by validation accuracy
  (epoch 11; val accuracy 0.9629). Training script included
  (`examples/train.py`).
- Verified weights in three formats: `model.safetensors`, the training
  checkpoint (`assets/pathocnn.pt`), and ONNX (`pathocnn.onnx`, parity
  vs PyTorch measured at 5.7e-06 max logit diff over 64 test images).
- Every headline metric independently reproduced from the shipped
  artifacts (`examples/verify_metrics.py`): test accuracy 0.8712, macro
  OvR AUC 0.9855, full 9x9 confusion matrix, per-class recalls — exact
  match with the training-time record. Weakest class documented on the
  card: cancer-associated stroma recall 0.3848.
- Inference API (`load_model`, `predict`, `predict_proba`, `preprocess`)
  with the exact training preprocessing contract.
- Demo (`examples/predict_demo.py`): one patch per class plus the first
  misclassified test patches, including the model's confident
  stroma/smooth-muscle → debris errors.
