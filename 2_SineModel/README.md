# Sine Model (2_SineModel)

## Abstract
This project demonstrates non-linear function approximation on edge devices by training a deep neural network to predict the trigonometric sine wave $y = \sin(x)$ across the interval $[0, 2\pi]$ using noisy synthetic samples. The model is converted using full INT8 Post-Training Quantization (PTQ) and serialized into an embedded C header byte array for memory-efficient microcontroller inference.

## Project Summary
- **Objective**: Approximate a non-linear periodic trigonometric function ($y = \sin(x)$) on resource-constrained embedded targets.
- **Input / Output**: Single scalar angle in radians ($x \in [0, 2\pi]$) mapped to a continuous sine amplitude prediction ($y \in [-1, 1]$).
- **Architecture**: A 3-layer Multilayer Perceptron (MLP) featuring an input layer, two hidden layers with 16 ReLU neurons each, and a single linear output neuron (321 trainable parameters).
- **Optimization & Quantization**: Trained across 250 epochs using the Adam optimizer, Mean Squared Error (MSE) loss, and Mean Absolute Error (MAE) tracking with an 80/20 train/validation split. Quantized to full INT8 using representative dataset calibration.
- **Artifacts in this Folder**:
  - `sine_model.py`: Python script for dataset synthesis, training, INT8 quantization, and C header generation.
  - `sine.tflite`: Quantized TensorFlow Lite FlatBuffer binary (3,424 bytes).
  - `sine_model.h.h`: C header file defining the exported model byte array (`sine_mod`) ready for embedded C/C++ compilation.
