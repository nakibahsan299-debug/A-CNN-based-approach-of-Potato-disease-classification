# Potato Leaf Disease Classification using CNN

A convolutional neural network (TensorFlow/Keras) that classifies potato leaf images into three classes: **Early Blight**, **Late Blight**, and **Healthy**. It reaches **99.59% accuracy** on the held-out test set.

## Overview

- **Task:** 3-class image classification of potato leaves
- **Data:** 2,134 RGB images, resized to 256×256
- **Split:** 80% train / 10% validation / 10% test (split by batch, batch size 32)
- **Framework:** TensorFlow / Keras, run in Google Colab

## Model

| Stage | Details |
|---|---|
| Preprocessing | Resize and rescale to [0, 1] |
| Augmentation | Random flip and random rotation (0.2) |
| Feature extractor | 6 × Conv2D (32, then 5 × 64 filters, 3×3, ReLU), each followed by 2×2 max pooling |
| Classifier | Flatten → Dense(64, ReLU) → Dense(3, softmax) |
| Training | Adam, sparse categorical cross-entropy, 50 epochs |

## Results

| Metric | Value |
|---|---|
| Final training accuracy | 98.3% |
| Final validation accuracy | 99.0% |
| Test accuracy | 99.6% |

The notebook also plots training and validation curves and shows sample predictions with confidence scores.

## Getting started

1. Open `Potato_disease_classification_using_CNN.ipynb` in Google Colab.
2. Put the dataset in Google Drive with one folder per class:
   ```
   PlantDataset/
   ├── Potato___Early_blight/
   ├── Potato___Late_blight/
   └── Potato___healthy/
   ```
3. Update the dataset path in the notebook (`/content/drive/MyDrive/PlantDataset`) if yours differs.
4. Run all cells.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy

## Dataset

Add a link to your dataset source here.

## Possible improvements

- Comparing against transfer learning (e.g. EfficientNetB0)
- Adding a confusion matrix and per-class precision/recall
- Spliting by image instead of by batch, and test on a larger held-out set
- Exporting the model and add a simple inference script or web demo

## License

Add your license here.
