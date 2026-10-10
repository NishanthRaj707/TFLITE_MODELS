# Sine Model: Edge AI Trigonometric Wave Approximation ($y = \sin(x)$)

A TinyML implementation demonstrating non-linear function approximation using a deep neural network, trained on noisy synthetic sine wave data, converted with full **INT8 Post-Training Quantization (PTQ)**, and exported into C byte array headers for microcontrollers.

---

## 📌 Project Overview

This subproject focuses on learning non-linear continuous mappings on embedded hardware:
- Generates 2,000 synthetic $(x, y)$ samples for $y = \sin(x)$ over $[0, 2\pi]$ with Gaussian noise.
- Splits data into 80% training and 20% validation sets.
- Trains a 3-layer Multilayer Perceptron (MLP) with 321 trainable parameters.
- Converts the trained model to an INT8 quantized FlatBuffer (`sine.tflite`).
- Exports the model to a C array header file (`sine_model.h.h` / `sine_model.h`).

---

## 📁 Directory Files

```
SineModel/
├── README.md          # Technical documentation (this file)
├── sine_model.py      # Complete training, validation, quantization, and C exporter script
├── sine.tflite        # Quantized INT8 TensorFlow Lite FlatBuffer (3,424 bytes)
└── sine_model.h.h     # Exported C byte array header (sine_mod)
```

> **Note on Header Filename**: In `sine_model.py`, the exporter function specifies `filename="sine_model.h"`, opening `f"{filename}.h"`, which writes to `sine_model.h.h`. In C/C++ projects, you can include `sine_model.h.h` directly or copy it to `sine_model.h`.

---

## 🧠 Python Pipeline Analysis (`sine_model.py`)

### 1. Synthetic Dataset Generation
- **Random Seed**: Fixed to `42` (`np.random.seed(42)`) for reproducible dataset generation.
- **Total Samples**: 2,000.
- **Input Domain ($x$)**: Drawn uniformly over the full trigonometric period $x \in [0, 2\pi] \approx [0.0, 6.28318]$.
- **Target Function ($y$)**: Sine function perturbed by additive Gaussian noise:
  $$y = \sin(x) + 0.1 \times \mathcal{N}(0, 1)$$
- **Train/Validation Partition**:
  - Training set: $80\% \implies 1,600 \text{ samples}$
  - Validation set: $20\% \implies 400 \text{ samples}$

```python
np.random.seed(42)
NUM_SAMPLES = 2000
x_values = np.random.uniform(0, 2 * np.pi, NUM_SAMPLES).astype(np.float32)
np.random.shuffle(x_values)
y_values = np.sin(x_values).astype(np.float32)
y_values += 0.1 * np.random.randn(*y_values.shape)

split = int(0.8 * NUM_SAMPLES)
x_train, x_validate = x_values[:split], x_values[split:]
y_train, y_validate = y_values[:split], y_values[split:]
```

### 2. Neural Network Architecture
Because $\sin(x)$ is non-linear, the model uses two hidden layers with non-linear activations:
- **Input Layer**: `keras.layers.Input(shape=(1,))` (1 scalar input angle in radians).
- **Hidden Layer 1**: `keras.layers.Dense(16, activation="relu")`.
- **Hidden Layer 2**: `keras.layers.Dense(16, activation="relu")`.
- **Output Layer**: `keras.layers.Dense(1)` (Linear identity output).

#### Parameter Calculation:
- **Hidden Layer 1**:
  $$\text{Weights} = 1 \times 16 = 16, \quad \text{Biases} = 16 \implies 32 \text{ parameters}$$
- **Hidden Layer 2**:
  $$\text{Weights} = 16 \times 16 = 256, \quad \text{Biases} = 16 \implies 272 \text{ parameters}$$
- **Output Layer**:
  $$\text{Weights} = 16 \times 1 = 16, \quad \text{Biases} = 1 \implies 17 \text{ parameters}$$
- **Total Trainable Parameters**:
  $$32 + 272 + 17 = 321 \text{ parameters}$$

### 3. Compilation & Training
- **Optimizer**: Adam (`optimizer="adam"`).
- **Loss Function**: Mean Squared Error (`loss="mse"`).
- **Monitoring Metric**: Mean Absolute Error (`metrics=["mae"]`).
- **Hyperparameters**: 250 epochs, batch size of 32.
- **Validation**: Evaluated against `(x_validate, y_validate)` on each epoch.

### 4. Full INT8 Post-Training Quantization (PTQ)
- **Representative Calibration**: Yields 100 sample training inputs from `x_train` to record activation dynamic ranges.
- **Operators**: Constrained to `TFLITE_BUILTINS_INT8`.
- **Input and Output Tensors**: Specified as signed 8-bit integers (`tf.int8`).

```python
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

### 5. C Header Generation
The script formats the 3,424 bytes of the binary FlatBuffer into comma-separated hex strings:
```c
const unsigned char sine_mod[] = {0x20, 0x00, 0x00, 0x00, ...};
const unsigned int sine_mod_len = 3424;
```

---

## 🚀 How to Run & Train

### 1. Install Dependencies
```bash
pip install tensorflow numpy
```

### 2. Execute Training Script
```bash
python sine_model.py
```

### 3. Expected Output
```text
Epoch 1/250
50/50 [==============================] - 0s 4ms/step - loss: 0.4820 - mae: 0.6120 - val_loss: 0.4010 - val_mae: 0.5480
...
Epoch 250/250
50/50 [==============================] - 0s 1ms/step - loss: 0.0125 - mae: 0.0890 - val_loss: 0.0118 - val_mae: 0.0854
Success: Exported 3424 bytes to model_data.h
```
*(Files generated: `sine.tflite` and `sine_model.h.h`)*

---

## 💻 Microcontroller Deployment Guide (C++ / TFLM)

Example integration into an embedded firmware project using TensorFlow Lite Micro:

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "sine_model.h.h" // or "sine_model.h"

// 1. Static Tensor Arena
constexpr int kTensorArenaSize = 3072;
uint8_t tensor_arena[kTensorArenaSize];

// 2. Load Model & Register Ops
const tflite::Model* model = tflite::GetModel(sine_mod);
tflite::MicroMutableOpResolver<1> resolver;
resolver.AddFullyConnected();

// 3. Initialize Interpreter
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
interpreter.AllocateTensors();

TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// 4. Quantize Input Angle x in Radians
float angle_radians = 3.14159f / 2.0f; // pi / 2
int8_t x_quant = (int8_t)(angle_radians / input->params.scale + input->params.zero_point);
input->data.int8[0] = x_quant;

// 5. Run Model
interpreter.Invoke();

// 6. Dequantize Output
int8_t y_quant = output->data.int8[0];
float sin_pred = (y_quant - output->params.zero_point) * output->params.scale;
// Expected sin_pred ~ 1.0f (since sin(pi / 2) = 1.0)
```

---

## 🐍 Standalone Python Verification Script

Validate accuracy and inference error against ground truth in Python:

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="sine.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_scale, in_zero_point = input_details["quantization"]
out_scale, out_zero_point = output_details["quantization"]

# Test angles: 0, pi/6, pi/4, pi/2, pi, 3pi/2, 2pi
test_angles = np.array([0.0, np.pi/6, np.pi/4, np.pi/2, np.pi, 1.5*np.pi, 2.0*np.pi], dtype=np.float32)

print(f"{'Input (rad)':<12} | {'Input (deg)':<12} | {'Pred sin(x)':<12} | {'True sin(x)':<12} | {'Abs Error':<10}")
print("-" * 70)

for angle in test_angles:
    q_in = np.round(angle / in_scale + in_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details["index"], np.array([[q_in]], dtype=np.int8))
    interpreter.invoke()
    
    q_out = interpreter.get_tensor(output_details["index"])[0][0]
    pred = (float(q_out) - out_zero_point) * out_scale
    true_val = float(np.sin(angle))
    error = abs(pred - true_val)
    
    print(f"{angle:<12.4f} | {np.degrees(angle):<12.1f} | {pred:<12.4f} | {true_val:<12.4f} | {error:<10.4f}")
```
