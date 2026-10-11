# TensorFlow Lite Micro (TinyML) Edge AI Repository

A curated collection of end-to-end TinyML neural network models built with **TensorFlow / Keras**, fully quantized using **INT8 Post-Training Quantization (PTQ)**, and exported into C byte array headers for deployment on resource-constrained microcontrollers (e.g., ESP32, STM32, Arduino Nano 33 BLE, Raspberry Pi Pico).

---

## 📌 Repository Overview

This repository demonstrates the complete lifecycle of embedded machine learning applications across different problem domains (regression and classification):
1. **Problem Formulation & Synthetic Data Generation** with realistic Gaussian sensor noise.
2. **Keras Sequential Model Architectures** designed for ultra-low memory footprints.
3. **Model Compilation & Training** using task-specific loss functions (MSE, Sparse Categorical Cross-Entropy).
4. **Full INT8 Post-Training Quantization (PTQ)** with representative dataset calibration.
5. **Serialization** into compact TensorFlow Lite FlatBuffers (`.tflite`).
6. **C Header Generation** (`.h`) containing static byte arrays for direct compilation into embedded C++ firmware.

---

## 📁 Projects in This Repository

This workspace contains 3 dedicated TinyML projects:

| Project Folder | Task | Description | Architecture | Quantization | Output Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[`1_Linearmodel/`](./1_Linearmodel)** | Regression | Fits the linear equation $y = 3x + 1$ | 1 Input $\to$ Dense(16) $\to$ 1 Output (49 params) | Full INT8 | `model.tflite`<br>`model_data.h` |
| **[`2_SineModel/`](./2_SineModel)** | Regression | Approximates non-linear sine wave $y = \sin(x)$ | 1 Input $\to$ Dense(16) $\to$ Dense(16) $\to$ 1 Output (321 params) | Full INT8 | `sine.tflite`<br>`sine_model.h.h` |
| **[`3_MultiClassification/`](./3_MultiClassification)** | Multi-Class Classification | 6-axis device orientation from 3D accelerometer data | 3 Inputs $\to$ Dense(16) $\to$ 6 Outputs (166 params) | Full INT8 | `mcc_model.tflite`<br>`mcc_model.h` |

---

## 📊 Architectural & Pipeline Comparison Matrix

| Attribute | `1_Linearmodel` | `2_SineModel` | `3_MultiClassification` |
| :--- | :--- | :--- | :--- |
| **Task Type** | 1D Linear Regression | 1D Non-Linear Regression | 3D Multi-Class Classification (6 classes) |
| **Target Function / Classes** | $y = 3x + 1 + \epsilon$ | $y = \sin(x) + \epsilon$ | +Z, -Z, +Y, -Y, +X, -X (Device orientations) |
| **Input Shape & Domain** | 1D: $x \in [-10.0, 10.0]$ | 1D: $x \in [0, 2\pi]$ | 3D: $[A_x, A_y, A_z]$ accelerometer vector |
| **Output Shape & Type** | 1 scalar (continuous) | 1 scalar (continuous) | 6 class logits (discrete state) |
| **Dataset Size** | 1,000 synthetic samples | 2,000 synthetic samples | 3,000 synthetic samples (500 / class) |
| **Model Topology** | `Input(1)` $\to$ `Dense(16, relu)` $\to$ `Dense(1)` | `Input(1)` $\to$ `Dense(16, relu)` $\to$ `Dense(16, relu)` $\to$ `Dense(1)` | `Input(3)` $\to$ `Dense(16, relu)` $\to$ `Dense(6)` |
| **Trainable Parameters** | **49 parameters** | **321 parameters** | **166 parameters** |
| **Loss Function** | Mean Squared Error (`mse`) | Mean Squared Error (`mse`) | Sparse Categorical Cross-Entropy |
| **Training Epochs** | 200 epochs | 250 epochs | 150 epochs |
| **Quantization Scheme** | Full Integer PTQ (`int8` I/O) | Full Integer PTQ (`int8` I/O) | Full Integer PTQ (`int8` I/O) |
| **TFLite FlatBuffer Size** | **2,232 bytes** | **3,424 bytes** | **2,496 bytes** |
| **C Array Identifier** | `g_model` (`g_model_len = 2232`) | `sine_mod` (`sine_mod_len = 3424`) | `g_model` (`g_model_len = 2496`) |
| **Exported Header** | `model_data.h` | `sine_model.h.h` | `mcc_model.h` |

---

## ⚡ Quantization Strategy (Full INT8 PTQ)

All models apply full integer post-training quantization to maximize runtime efficiency and minimize RAM consumption on microcontrollers without floating-point units:

- **Footprint Reduction**: Quantizes 32-bit floating-point weights and activation tensors to signed 8-bit integers (`int8`), achieving an approximate 4x memory reduction.
- **Representative Calibration**: Feeds 100 sample training inputs through `tf.lite.TFLiteConverter` to accurately map min/max dynamic ranges.
- **Affine Quantization Equation**:
  $$q = \text{round}\left(\frac{r}{S}\right) + Z$$
  where $r$ is the real float value, $q \in [-128, 127]$ is the quantized integer, $S$ is the scale factor, and $Z$ is the zero-point.

---

## 📜 Repository Structure

```
.
├── README.md                   # Repository overview and cross-project comparison (this file)
├── 1_Linearmodel/              # Project 1: Baseline linear regression
│   ├── README.md               # Abstract and summary for 1_Linearmodel
│   ├── main.py                 # Training, INT8 PTQ quantization, and C export script
│   ├── model.tflite            # INT8 FlatBuffer model (2,232 bytes)
│   └── model_data.h            # Exported C byte array header (g_model)
├── 2_SineModel/                # Project 2: Non-linear trigonometric approximation
│   ├── README.md               # Abstract and summary for 2_SineModel
│   ├── sine_model.py           # Training, validation, INT8 PTQ, and C export script
│   ├── sine.tflite             # INT8 FlatBuffer model (3,424 bytes)
│   └── sine_model.h.h          # Exported C byte array header (sine_mod)
└── 3_MultiClassification/      # Project 3: 6-axis spatial orientation classification
    ├── README.md               # Abstract and summary for 3_MultiClassification
    ├── mcc.py                  # Training, INT8 PTQ, and C export script
    ├── mcc_model.tflite        # INT8 FlatBuffer model (2,496 bytes)
    └── mcc_model.h             # Exported C byte array header (g_model)
```

---

## 📄 License
This repository is released under the **MIT License**.
