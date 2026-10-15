# 1D CNN Gesture / Motion Classifier (5_CNN_1D)

## Abstract

This project implements an edge AI gesture recognition model using a 1D Convolutional Neural Network (CNN) to classify raw 3-axis accelerometer time-series windows into three motion classes: IDLE, WAVE, and PUNCH. Unlike the earlier MLP-based models in this repository, the Conv1D architecture is able to extract temporal features directly from the raw sensor signal without manual feature engineering. The trained model is converted to a fully INT8 quantized TensorFlow Lite FlatBuffer and serialized into a C header file for real-time inference on microcontrollers.

## Project Summary

- **Objective**: Classify a 50-sample, 3-axis (Ax, Ay, Az) accelerometer window into one of three gesture classes — IDLE, WAVE, or PUNCH — directly from raw time-series data.
- **Input / Output**: A 2D tensor of shape `(50, 3)` representing 1.0 second of IMU data at 50 Hz, mapped to 3 class logits via Sparse Categorical Cross-Entropy.
- **Architecture**: A compact 1D Convolutional Neural Network consisting of a `Conv1D(8, kernel=3, relu)` feature extraction layer, a `MaxPooling1D(2)` downsampling layer, a `GlobalAveragePooling1D` aggregation layer, and a `Dense(3)` classification head.
- **Dataset**: 1,200 synthetic IMU windows — 400 samples per class, each 50 timesteps × 3 channels:
  - **Class 0 — IDLE**: Static flat orientation with gravity on Az (+1.0g) and low-amplitude thermal sensor noise (σ = 0.05).
  - **Class 1 — WAVE**: Smooth low-frequency sinusoidal oscillation on Ax and Ay (1.8–2.5 Hz random variation) with a partial gravity component on Az.
  - **Class 2 — PUNCH**: Random-position forward acceleration spike (+3.0g) followed by a sharp braking deceleration (-2.0g) on Ax, simulating an impact transient.
- **Optimization & Quantization**: Trained over 150 epochs using the Adam optimizer with Sparse Categorical Cross-Entropy loss and a 20% validation split. Fully quantized to INT8 using 100-sample representative dataset calibration.
- **Bugs Fixed**:
  - `X_data.shape` → `x_data.shape` (NameError after shuffle lowercased the variable).
  - `yield x_data[i:i+1]` → `yield [x_data[i:i+1]]` (TFLite representative dataset requires a list-wrapped tensor for non-MLP models).
  - Export filenames `"binary_model"` / `"b_model"` → `"cnn_model"` / `"cnn_model"` (copy-paste residual from `4_BinaryClassification`).
- **Artifacts in this Folder**:
  - `cnn.py`: Python script for synthetic IMU data generation, CNN training, INT8 quantization, and C header export.
  - `cnn.tflite`: Quantized TensorFlow Lite FlatBuffer binary.
  - `cnn_model.h`: C header file exporting the model as a static byte array (`cnn_model`) for embedded C/C++ firmware.
