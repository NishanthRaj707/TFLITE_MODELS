# TensorFlow Lite Micro (TinyML) Edge AI Repository

A curated collection of end-to-end TinyML neural network models built with **TensorFlow / Keras**, fully quantized using **INT8 Post-Training Quantization (PTQ)**, and exported into C byte array headers for deployment on resource-constrained microcontrollers (e.g., ESP32, STM32, Arduino Nano 33 BLE, Raspberry Pi Pico).

---

## 📌 Repository Overview

This repository demonstrates the complete lifecycle of embedded machine learning applications:
1. **Mathematical Formulation & Synthetic Data Generation** with Gaussian noise.
2. **Keras Sequential Model Architecture** designed for ultra-low memory footprints.
3. **Model Compilation & Training** using Mean Squared Error (MSE) loss.
4. **Full INT8 Post-Training Quantization (PTQ)** with representative dataset calibration.
5. **Serialization** into TensorFlow Lite FlatBuffers (`.tflite`).
6. **C Header Generation** (`.h`) containing static byte arrays for direct compilation into embedded C++ firmware.

```
┌─────────────────────────────────┐
│  1. Synthetic Dataset Creation  │ (Uniform distribution + Gaussian noise)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│    2. Keras Sequential Model    │ (Dense layers with ReLU activations)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│     3. Model Compilation        │ (Adam Optimizer, MSE Loss, Epochs 200-250)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 4. Full INT8 PTQ Quantization   │ (Representative dataset calibration, int8 I/O)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  5. TFLite FlatBuffer Export    │ (.tflite binary file)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   6. C Byte Array Generation    │ (const unsigned char model[] in .h header)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 7. MCU Deployment (TFLM/ESP32)  │ (Tensor Arena allocation & fixed-point inference)
└─────────────────────────────────┘
```

---

## 📁 Projects in This Repository

This workspace is structured into dedicated subprojects targeting different mathematical functions:

| Project Folder | Description | Model Architecture | Quantization | Output Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **[`Linearmodel/`](./Linearmodel)** | Fits the linear equation $y = 3x + 1$ | 1 Input $\to$ 16 Hidden $\to$ 1 Output (49 params) | Full INT8 | `model.tflite`<br>`model_data.h` |
| **[`SineModel/`](./SineModel)** | Approximates non-linear sine wave $y = \sin(x)$ | 1 Input $\to$ 16 Hidden $\to$ 16 Hidden $\to$ 1 Output (321 params) | Full INT8 | `sine.tflite`<br>`sine_model.h.h` / `sine_model.h` |

### Architectural & Pipeline Comparison Matrix

| Attribute | `Linearmodel` (`main.py`) | `SineModel` (`sine_model.py`) |
| :--- | :--- | :--- |
| **Target Function** | Linear Regression: $y = 3x + 1 + \epsilon$ | Trigonometric Wave: $y = \sin(x) + \epsilon$ |
| **Input Domain** | $x \in [-10.0, 10.0]$ | $x \in [0, 2\pi] \approx [0.0, 6.28318]$ |
| **Noise Distribution** | $\mathcal{N}(0, 0.1^2)$ | $0.1 \times \mathcal{N}(0, 1)$ |
| **Dataset Size** | 1,000 samples (100% training) | 2,000 samples (80% train: 1,600 / 20% val: 400) |
| **Random Seed** | Unseeded (`np.random.uniform`) | Seeded (`np.random.seed(42)`) |
| **Hidden Layers** | 1 layer: `Dense(16, activation="relu")` | 2 layers: `Dense(16, relu)` $\to$ `Dense(16, relu)` |
| **Output Layer** | `Dense(1)` (Linear identity) | `Dense(1)` (Linear identity) |
| **Trainable Parameters** | **49 parameters** | **321 parameters** |
| **Loss & Metrics** | Loss: `MSE` | Loss: `MSE`, Metric: `MAE` |
| **Training Schedule** | 200 Epochs, Batch Size 32 | 250 Epochs, Batch Size 32, Validation Data |
| **Quantization Scheme** | Full Integer Post-Training Quantization (`INT8`) | Full Integer Post-Training Quantization (`INT8`) |
| **Representative Dataset** | 100 samples yielded individually | 100 samples yielded individually |
| **Inference Types** | Input: `int8`, Output: `int8` | Input: `int8`, Output: `int8` |
| **TFLite FlatBuffer** | `model.tflite` (2,232 bytes) | `sine.tflite` (3,424 bytes) |
| **Exported C Array** | `const unsigned char g_model[]` | `const unsigned char sine_mod[]` |
| **Exported Length** | `g_model_len = 2232` | `sine_mod_len = 3424` |
| **Target Header File** | `model_data.h` | `sine_model.h.h` / `sine_model.h` |

---

## ⚡ Quantization Strategy (Full INT8 PTQ)

Both projects apply full integer post-training quantization to maximize performance on edge hardware lacking hardware Floating-Point Units (FPUs):

- **Memory Reduction**: Converts 32-bit floating-point weights and biases into 8-bit integers (`int8`), achieving a ~4x reduction in model size.
- **Representative Dataset**: Feeds 100 sample training inputs through `tf.lite.TFLiteConverter` to record the dynamic ranges (min/max activation values).
- **Affine Quantization Equation**:
  $$q = \text{round}\left(\frac{r}{S}\right) + Z$$
  where $r$ is the real float value, $q \in [-128, 127]$ is the quantized 8-bit integer, $S$ is the scale factor, and $Z$ is the integer zero-point.
- **Dequantization Equation**:
  $$r = S \times (q - Z)$$

---

## 🛠️ Prerequisites & Environment Setup

### 1. Python Environment
- Python 3.9 – 3.12
- TensorFlow 2.x
- NumPy

Install the required packages:
```bash
pip install tensorflow numpy
```

### 2. Embedded Target Support
The exported C byte arrays are compatible with:
- **TensorFlow Lite Micro (TFLM)**
- **ESP-IDF** (with `espressif__esp-tflite-micro`)
- **Arduino IDE** (with `Arduino_TensorFlowLite` library)
- **STM32Cube.AI**

---

## 🚀 Quick Execution Guide

### Training the Linear Model
```bash
cd Linearmodel
python main.py
```
*Generated files: `model.tflite` and `model_data.h`.*

### Training the Sine Model
```bash
cd SineModel
python sine_model.py
```
*Generated files: `sine.tflite` and `sine_model.h.h` / `sine_model.h`.*

---

## 📜 Repository Structure

```
.
├── README.md               # Repository-wide overview and documentation (this file)
├── Linearmodel/            # Linear regression TinyML project (y = 3x + 1)
│   ├── README.md           # Complete technical documentation for Linear Model
│   ├── main.py             # Training, INT8 PTQ quantization, and C header exporter
│   ├── model.tflite        # Generated INT8 FlatBuffer model (2,232 bytes)
│   └── model_data.h        # Exported C byte array (g_model)
└── SineModel/              # Sine wave approximation TinyML project (y = sin(x))
    ├── README.md           # Complete technical documentation for Sine Model
    ├── sine_model.py       # Training, validation, INT8 PTQ, and C exporter
    ├── sine.tflite         # Generated INT8 FlatBuffer model (3,424 bytes)
    └── sine_model.h.h      # Exported C byte array (sine_mod)
```

---

## 📄 License
This repository is released under the **MIT License**.
