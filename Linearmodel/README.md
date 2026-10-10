# Linear Model: Edge AI Linear Regression ($y = 3x + 1$)

A TinyML implementation demonstrating synthetic data generation, neural network training, full **INT8 Post-Training Quantization (PTQ)**, and C header generation for predicting the linear function $y = 3x + 1$.

---

## 📌 Project Overview

This subproject implements a compact regression model suitable for baseline validation on edge microcontrollers. It covers:
- Synthetic 1D regression data creation with Gaussian noise.
- Training a 2-layer Multilayer Perceptron (MLP) in TensorFlow/Keras.
- Full INT8 quantization using a representative calibration dataset.
- Exporting the resulting model into a binary FlatBuffer (`model.tflite`) and a C byte array (`model_data.h`).

---

## 📁 Directory Files

```
Linearmodel/
├── README.md       # Technical documentation (this file)
├── main.py         # Complete training, quantization, and C export script
├── model.tflite    # Quantized INT8 TensorFlow Lite FlatBuffer (2,232 bytes)
└── model_data.h    # Embedded C byte array header (g_model)
```

---

## 🧠 Python Pipeline Analysis (`main.py`)

### 1. Synthetic Dataset Generation
The script generates 1,000 synthetic 1D data points:
- **Input Features ($x_{\text{train}}$)**: Sampled uniformly over $x \in [-10.0, 10.0]$ with `dtype=float32`.
- **Target Labels ($y_{\text{train}}$)**: Evaluated using the linear formula with added Gaussian noise:
  $$y = 3x + 1 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.1^2)$$

```python
x_train = np.random.uniform(-10, 10, size=(1000, 1)).astype(np.float32)
y_train = (3 * x_train + 1) + np.random.normal(0, 0.1, size=(1000, 1)).astype(np.float32)
```

### 2. Neural Network Architecture
The network is built using the Keras Sequential API:
- **Input Layer**: `keras.layers.Input(shape=(1,))` (1 scalar value).
- **Hidden Layer**: `keras.layers.Dense(16, activation="relu")` (16 hidden units, Rectified Linear Unit activation).
- **Output Layer**: `keras.layers.Dense(1)` (Linear activation for continuous regression).

#### Parameter Calculation:
- **Hidden Layer**:
  $$\text{Weights} = 1 \times 16 = 16, \quad \text{Biases} = 16 \implies 32 \text{ parameters}$$
- **Output Layer**:
  $$\text{Weights} = 16 \times 1 = 16, \quad \text{Biases} = 1 \implies 17 \text{ parameters}$$
- **Total Trainable Parameters**:
  $$32 + 17 = 49 \text{ parameters}$$

### 3. Compilation & Training
- **Optimizer**: Adam (`optimizer="adam"`, default learning rate $\eta = 0.001$).
- **Loss Function**: Mean Squared Error (`loss="mse"`):
  $$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$
- **Training Parameters**: 200 epochs, batch size of 32.

### 4. Full INT8 Post-Training Quantization (PTQ)
To enable fast fixed-point arithmetic on microcontrollers without hardware FPUs:
- **Representative Dataset**: Supplies 100 sample inputs from `x_train` to calibrate the dynamic range of activations.
- **Target Operations**: Restricted strictly to `TFLITE_BUILTINS_INT8`.
- **Inference Input/Output Types**: Set to `tf.int8`.

```python
def rep_dataset():
    for i in range(100):
        yield [x_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

### 5. C Header Generation
The script converts raw binary bytes into a C array of hexadecimal literals and writes it to `model_data.h`:
```c
const unsigned char g_model[] = {0x1c, 0x00, 0x00, 0x00, ...};
const unsigned int g_model_len = 2232;
```

---

## 🚀 How to Run & Train

### 1. Install Dependencies
```bash
pip install tensorflow numpy
```

### 2. Execute Training Script
```bash
python main.py
```

### 3. Expected Output
```text
Starting the training of the model
Epoch 1/200
32/32 [==============================] - 0s 1ms/step - loss: 104.3821
...
Epoch 200/200
32/32 [==============================] - 0s 1ms/step - loss: 0.0098
Initialising the converter
Success: Saved model.tflite
Success: Exported 2232 bytes to model_data.h
```

---

## 💻 Microcontroller Deployment Guide (C++ / TFLM)

To run this model on an embedded device (e.g., ESP32 or Arduino), incorporate `model_data.h` with TensorFlow Lite Micro:

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// 1. Allocate Static Memory Arena
constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// 2. Load Model & Op Resolver
const tflite::Model* model = tflite::GetModel(g_model);
tflite::MicroMutableOpResolver<1> resolver;
resolver.AddFullyConnected();

// 3. Initialize Interpreter
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
interpreter.AllocateTensors();

TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// 4. Quantize Float Input to INT8
float x_val = 2.0f;
int8_t x_quantized = (int8_t)(x_val / input->params.scale + input->params.zero_point);
input->data.int8[0] = x_quantized;

// 5. Invoke Inference
interpreter.Invoke();

// 6. Dequantize INT8 Output to Float
int8_t y_quantized = output->data.int8[0];
float y_pred = (y_quantized - output->params.zero_point) * output->params.scale;
// Expected y_pred ~ 7.0f (since 3 * 2.0 + 1 = 7.0)
```

---

## 🐍 Standalone Python Verification Script

To test the generated `.tflite` model directly in Python:

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_scale, in_zero_point = input_details["quantization"]
out_scale, out_zero_point = output_details["quantization"]

test_x = np.array([[2.0], [5.0], [-3.0]], dtype=np.float32)

for x in test_x:
    # Quantize input
    q_in = np.round(x / in_scale + in_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details["index"], q_in)
    interpreter.invoke()
    
    # Dequantize output
    q_out = interpreter.get_tensor(output_details["index"])
    y_pred = (q_out.astype(np.float32) - out_zero_point) * out_scale
    y_true = 3.0 * x[0] + 1.0
    print(f"Input: {x[0]:5.1f} | Pred: {y_pred[0][0]:6.2f} | Ground Truth: {y_true:6.2f}")
```
