

# Object Detection Assignment

This repository contains the implementation and training code for an object detection model built as part of an AI internship assignment. The model is based on adding detection layers on top of a pre-trained CNN backbone and trained on a custom dataset with XML annotations.


## Overview

The objective is to develop an object detection model leveraging a ResNet-based backbone integrated into an SSD (Single Shot MultiBox Detector) framework. The model is trained to detect persons using a dataset containing XML annotations. The assignment covers:

* Selecting a suitable CNN backbone.
* Designing and integrating detection heads.
* Preparing a training pipeline with data augmentation.
* Evaluating model performance with mAP metrics.

---

## Project Structure

* `data/` — Dataset files and XML annotations.
* `model.py` — Defines the SSD model with ResNet backbone and detection heads.
* `train.py` — Training loop, optimizer, scheduler, and checkpointing.
* `eval.py` — Evaluation script reporting mAP\@0.5 and mAP\@0.5:0.95 metrics.
* `custom_utils.py` — Utility functions for dataset handling and preprocessing.
* `config.py` — Configuration file for hyperparameters and settings.
* `outputs/` — Directory to save model checkpoints and inference results.

---

## Model Architecture

* Backbone: Pretrained ResNet50, feature extractor layers adapted for multi-scale SSD features.
* Extra layers: Convolutional layers added for additional feature maps.
* Detection head: SSDHead that outputs class and bounding box predictions.
* Anchor boxes: DefaultBoxGenerator with predefined scales and aspect ratios.

---

## Dataset

* Dataset used: [Person Detection with XML Annotations](https://www.kaggle.com/datasets/sovitrath/person-detection-with-xml-annotations).
* Dataset format: Images with Pascal VOC style XML annotations.
* Data augmentation includes random horizontal flips and normalization.

---

## Training

* Framework: PyTorch.
* Optimizer: Adam with learning rate scheduling.
* Batch size, learning rate, epochs, and other hyperparameters configured in `config.py`.
* Training progress logged with tqdm progress bars.

---

## Evaluation

* Metrics: mAP\@0.5, mAP\@0.5:0.95, precision, and recall calculated with TorchMetrics.
* Validation warnings for many detections are handled gracefully.
* Results saved to logs for tracking.

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/muthumeenal12/object-detection-assignment.git
   cd object-detection-assignment
   ```

2. Install dependencies (recommended in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset inside `data/` and update paths in `config.py`.

4. Train the model:

   ```bash
   python train.py
   ```

5. Evaluate the model:

   ```bash
   python eval.py
   ```



---

## Results

* Achieved mAP\@0.5: **66.1%**
* Achieved mAP\@0.5:0.95: **26.0%**
* Noted challenges include balancing detection head size and anchor box tuning.

---

Report link: [doc](https://docs.google.com/document/d/1z5fM_aGmRDLPQaHKJAk1BUT19ZBCQ2UpIxHSIrAWrow/edit?usp=sharing)
