# Linear Model (1_Linearmodel)

## Abstract
This project implements a baseline TinyML regression model designed to learn the linear relationship $y = 3x + 1$ from synthetic data corrupted with Gaussian noise. The trained neural network is converted into an ultra-compact, fully quantized INT8 TensorFlow Lite model and exported as a C header file for direct deployment onto resource-constrained edge microcontrollers.

## Project Summary
- **Objective**: Learn and predict a continuous linear function ($y = 3x + 1$) on edge hardware.
- **Input / Output**: Single scalar input feature ($x \in [-10, 10]$) mapped to a single scalar regression output ($y$).
- **Architecture**: A compact Multilayer Perceptron (MLP) consisting of an input layer, one hidden layer with 16 ReLU neurons, and a single linear output neuron (49 trainable parameters).
- **Optimization & Quantization**: Trained using the Adam optimizer with Mean Squared Error (MSE) loss, followed by full integer Post-Training Quantization (INT8 PTQ) calibrated via a representative dataset.
- **Artifacts in this Folder**:
  - `main.py`: Python script handling synthetic data generation, model training, INT8 quantization, and C array serialization.
  - `model.tflite`: Quantized TensorFlow Lite model binary (2,232 bytes).
  - `model_data.h`: C header file containing the model as a static byte array (`g_model`) for microcontroller firmware integration.
