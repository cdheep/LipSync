**# Lip-to-Speech**

Convert silent lip-movement videos into text-based speech using deep learning.

![Screenshot 2025-04-14 205935](https://github.com/user-attachments/assets/6390b76d-600d-4430-9682-eebedc224080)


Lip-to-Speech is a deep learning pipeline that transforms silent video of lip movements into text-based speech. It uses a 3D convolutional feature extractor, bidirectional LSTM, and CTC loss to align sequence outputs without frame‑level labels. Trained on mouth ROI frames, it outputs transcriptions and synthesized speech accurately in near real-time.


---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Model Architecture](#model-architecture)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Training](#training)  
- [Inference](#inference)  
- [Results](#results)  
- [Directory Structure](#directory-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)  

---

## Project Overview

Lip2Speech implements a lip-reading neural network that takes short video clips of silent speech (lip movements) and generates the corresponding text transcription. The core pipeline:

1. **Preprocessing:**  
   - Download and unzip pre-segmented video frames.  
   - Resize and normalize each frame to 46×140 and stack into 75-frame clips.  
2. **Model:**  
   - A 3D-Convolutional feature extractor + Bi-LSTM + Time-Distributed Dense layers.  
   - CTC-based loss for sequence alignment without frame-level labels.  
3. **Training:**  
   - Adam optimizer, custom learning-rate scheduler, and CTC loss.  
4. **Inference:**  
   - Load trained weights, run on test clips, decode predictions.  

---

## Features

- **End-to-End Pipeline**: From raw frames to text.  
- **CTC-Loss**: No need for exact frame-to-phoneme alignment.  
- **GPU-Accelerated**: Leverages TensorFlow GPU support.  
- **Callbacks**: Periodic sample generation and learning-rate scheduling.  

---

## Model Architecture

| Layer Type           | Output Shape              | Parameters       |
| -------------------- | ------------------------- | ---------------- |
| `Conv3D(128, 3×3×3)` | `(75, 46, 140, 128)`      | …                |
| `BatchNorm + ReLU`   | `(75, 46, 140, 128)`      | …                |
| `SpatialDropout3D`   | `(75, 46, 140, 128)`      | —                |
| `Conv3D(256, 3×3×3)` | `(75, 46, 140, 256)`      | …                |
| `…`                  |                           |                  |
| `TimeDistributed(…)` | `(75, vocab_size)`        | …                |
| `CTC Decoder`        | `Sequence of characters`  | —                |

> **Input shape:** `(75 frames, 46×140 pixels, 1 channel)`  
> **Output:** Sequence of character probabilities per frame.  

_For full details see [LipNet.ipynb](./LipNet.ipynb)._

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- OpenCV, NumPy, Matplotlib, ImageIO, gdown

## Results

- For detailed training curves, evaluation metrics (WER/CER) and sample outputs, see the [LipNet notebook](./LipNet.ipynb).  
- Listen to the generated speech in [predicted_Speech.mp3](./predicted_speech.mp3).
- ![Screenshot 2025-04-15 111147](https://github.com/user-attachments/assets/a51e2483-6757-4b90-bed9-6b720782ed34)





