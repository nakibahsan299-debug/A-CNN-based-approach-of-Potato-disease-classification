# Potato Leaf Disease Classification using CNN

This project builds a convolutional neural network to detect common potato leaf diseases from images. It classifies leaves as Early Blight, Late Blight, or Healthy using 2,134 images resized to 256×256. The pipeline covers data loading with image_dataset_from_directory, an 80/10/10 train/validation/test split, caching and prefetching, rescaling, and flip and rotation augmentation. The model has six convolution and pooling blocks followed by a dense classifier, trained with Adam for 50 epochs. It reaches about 98.3% training, 99.0% validation, and 99.6% test accuracy. The notebook runs in Google Colab and includes training curves and sample predictions with confidence scores.

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


## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy


## Possible improvements

- Comparing against transfer learning (e.g. EfficientNetB0)
- Adding a confusion matrix and per-class precision/recall
- Spliting by image instead of by batch, and test on a larger held-out set
- Exporting the model and add a simple inference script or web demo

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
