# Multi-Class Orientation Classification (3_MultiClassification)

## Abstract
This project implements an edge AI multi-class classification model that determines device spatial orientation along 6 cardinal directions based on 3-axis accelerometer readings. Using synthetic gravity acceleration vectors with additive Gaussian sensor noise, the model is trained, converted into a fully quantized INT8 TensorFlow Lite model, and exported as a C header file for real-time edge inference on microcontrollers.

## Project Summary
- **Objective**: Classify 3D spatial orientation into one of 6 discrete directional states (+Z, -Z, +Y, -Y, +X, -X) using simulated 3-axis inertial measurement unit (IMU) sensor data.
- **Input / Output**: 3-dimensional acceleration vector $[A_x, A_y, A_z]$ mapped to 6 classification logits representing the 6 primary spatial orientations.
- **Architecture**: A lightweight Multilayer Perceptron (MLP) consisting of a 3-dimensional input layer, a hidden dense layer with 16 ReLU units, and a 6-unit output layer (166 trainable parameters).
- **Optimization & Quantization**: Trained over 150 epochs using the Adam optimizer with Sparse Categorical Cross-Entropy loss on 3,000 synthetic sensor readings (500 per orientation class) with a 20% validation split. Fully quantized to INT8 precision using representative dataset calibration.
- **Artifacts in this Folder**:
  - `mcc.py`: Python script covering synthetic accelerometer data generation, model training, INT8 quantization, and C header generation.
  - `mcc_model.tflite`: Quantized TensorFlow Lite FlatBuffer binary (2,496 bytes).
  - `mcc_model.h`: C header file exporting the model as a static byte array (`g_model`) for embedded firmware.
