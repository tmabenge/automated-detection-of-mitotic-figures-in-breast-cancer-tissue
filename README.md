# Mitotic Figure Detection in Canine Cutaneous Mast Cell Tumors (DICOM WSIs)

This repository contains code for training and evaluating a deep learning model to detect mitotic figures in canine cutaneous mast cell tumor (CCMCT) whole slide images (WSIs) stored in DICOM format. The model is trained using PyTorch/XLA on TPUs.

## Dataset

The project uses the "Mitosis WSI CCMCT Training Set" dataset available on Kaggle: [https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-training-set](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-training-set)

This dataset includes:
- 32 DICOM WSIs of canine cutaneous mast cell tumors.
- Annotations of mitotic figures, neoplastic mast cells, inflammatory granulocytes, and mitotic figure look-alikes in a SQLite database.

## Model

The project uses a Faster R-CNN model with a ResNet-50 backbone pre-trained on ImageNet. You can easily adapt the code to use other object detection models from the `torchvision.models.detection` module.

## Training

The model is trained using PyTorch/XLA on a TPU VM v3-8. The training code is optimized for memory efficiency by processing tiles (frames) from the DICOM WSIs individually and using a `DataLoader` with multiple worker processes and pinned memory.

## Evaluation

The model is evaluated on a held-out test set using standard object detection metrics such as mean Average Precision (mAP), precision, and recall.

## Visualization

The `visualize_predictions` function allows you to visualize the model's predictions by drawing bounding boxes on the image patches.

## Repository Structure
