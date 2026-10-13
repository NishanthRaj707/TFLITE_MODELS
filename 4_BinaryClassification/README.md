# Binary Classification (4_BinaryClassification)

## Abstract

This project implements an edge AI binary classification model that determines whether a 3-axis accelerometer reading represents a positive or negative spatial orientation. Using synthetic gravity vectors along the three cardinal axes with additive Gaussian sensor noise, the model is trained, converted into a fully quantized INT8 TensorFlow Lite model, and exported as a C header file for real-time edge inference on microcontrollers.

## Project Summary

- **Objective**: Classify a 3D accelerometer reading as either a positive orientation (+X, +Y, +Z) or a negative orientation (-X, -Y, -Z), producing a single binary output.
- **Input / Output**: 3-dimensional acceleration vector $[A_x, A_y, A_z]$ mapped to a single sigmoid output (0 = negative orientation, 1 = positive orientation).
- **Architecture**: A Multilayer Perceptron (MLP) with a 3-dimensional input layer, a 16-unit ReLU hidden layer, a 6-unit ReLU hidden layer, and a 1-unit sigmoid output layer.
- **Dataset**: 3,000 synthetic accelerometer samples — 1,500 positive class samples (500 per +X, +Y, +Z axis) and 1,500 negative class samples (500 per -X, -Y, -Z axis), each corrupted with Gaussian noise (mean=0, std=0.15).
- **Optimization & Quantization**: Trained over 150 epochs using the Adam optimizer with Binary Cross-Entropy loss and a 20% validation split. Fully quantized to INT8 precision using representative dataset calibration.
- **Artifacts in this Folder**:
  - `binary_model.py`: Python script for synthetic data generation, model training, INT8 quantization, and C header export.
  - `binary_model.tflite`: Quantized TensorFlow Lite FlatBuffer binary.
  - `binary_model.h`: C header file exporting the model as a static byte array (`b_model`) for embedded firmware.
