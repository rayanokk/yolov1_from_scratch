
## YOLOv1 From Scratch
A from-scratch implementation of YOLOv1 in PyTorch, built as a hands-on introduction to modern computer vision and object detection 

This project implements the YOLOv1 (*You Only Look Once*) object detection architecture introduce by Redmon et al. in 2015.

The goal is not simply to reproduce YOLOv1, but to provide a clear and modular implementation for understanding how an object detection model works from the ground up.

![YOLOv1 Architecture](https://miro.medium.com/1*tWAIXZ7T-pGAeWYBlnEOYw.png)
## Why this project ?

Object detection combines several fundamental concepts of computer vision:

 - Image representation and convolutional neural networks    
 - Bounding-box regression   
 - Objectness prediction  
 - Multi-class classification  
 - Intersection over Union   
 - Post-processing and Non-Maximum Suppression  
 - Real-time inference

Implementing these components from scratch makes the internal mechanics of an object detector much easier to understand.

This project is therefore intended as a learning-oriented introduction to computer vision and object detection, rather than a production-ready detection framework. 
## Features

####  YOLOv1 Architecture
![](https://figures.semanticscholar.org/f8e79ac0ea341056ef20f2616628b3e964764cfd/3-Figure3-1.png)
A modular implementation of the original **YOLOv1** architecture, including its convolutional backbone and detection head.

The model predicts an output tensor of:

**`S × S × (C + 5B)`**

where:

- **`S = 7`** — grid size
- **`B = 2`** — bounding boxes per grid cell
- **`C = 20`** — number of classes

Therefore, for PASCAL VOC:

**`7 × 7 × 30`**

predictions are produced for each image.

### Target Encoding 
Ground-truth bounding boxes are converted into the YOLO grid representation.

Each image is divided into a **`7 × 7`** grid, where each cell is responsible for predicting:

- bounding-box coordinates
- object confidence
- class probabilities

### YOLOv1 Loss
![](https://media.geeksforgeeks.org/wp-content/uploads/20250703152944325454/YOLO-loss-function.webp)
The implementation follows the multi-component loss described in Equation 3 of the original YOLO paper.

It combines:

- Bounding-box localization loss
- Width and height loss using square-root coordinates
- Object confidence loss
- No-object confidence loss
- Classification loss

The loss computation is implemented in a vectorized way and includes numerical stabilizations for improved training stability.

### IoU & Bounding Boxes

The project includes the computation of **Intersection over Union** (IoU) for bounding boxes represented as:


**``(x, y, w, h)``**


IoU is used both during training and evaluation to measure the overlap between predicted and ground-truth boxes.

### Non-Maximum Suppression

Predictions are decoded from the YOLO grid representation into image coordinates and filtered using **Non-Maximum Suppression** (NMS).

This removes redundant detections corresponding to the same object

### Real-Time Webcam Detection

The project also includes a simple OpenCV webcam demo showing how a trained detector can be integrated into a real-time computer vision pipeline.
##  Project Structure

```text
YOLOv1-from-scratch/
│
├──  model.py
│   └── YOLOv1 architecture and CNN blocks
│
├──  loss.py
│   └── YOLOv1 loss function and IoU computation
│
├──  dataset.py
│   └── PyTorch YOLO dataset
│
├──  data.py
│   └── PASCAL VOC data parsing
│
├──  target_encoder.py
│   └── Ground-truth → YOLO grid encoding
│
├──  postprocess.py
│   └── Bounding-box decoding, NMS and visualization
│
├──  metrics.py
│   └── Evaluation metrics
│
├──  webcam_demo.py
│   └── Real-time webcam inference
│
└──  README.md
```
## Documentation

[You Only Look Once:
Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)


## Badges


[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)


