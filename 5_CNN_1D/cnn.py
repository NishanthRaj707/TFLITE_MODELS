import tensorflow as tf
from tensorflow import keras
import numpy  as np

WINDOW_SIZE = 50   # 50 time steps (1.0 second at 50 Hz)
CHANNELS = 3      # Ax, Ay, Az
SAMPLES_PER_CLASS = 400

# Classes:
# 0: IDLE   - Static baseline gravity on Az (+1.0g) + thermal sensor noise
# 1: WAVE   - Smooth low-frequency sinusoidal motion on Ax (~2 Hz)
# 2: PUNCH  - Violent transient acceleration spike on Ax and quick braking

X_list, y_list = [], []
time_steps = np.linspace(0, 1.0, WINDOW_SIZE)

# Class 0: IDLE
for _ in range(SAMPLES_PER_CLASS):
    noise = np.random.normal(0, 0.05, size=(WINDOW_SIZE, CHANNELS))
    sample = noise + np.array([0.0, 0.0, 1.0])  # Stationary flat
    X_list.append(sample)
    y_list.append(0)

# Class 1: WAVE (Periodic oscillation on X and Y)
for _ in range(SAMPLES_PER_CLASS):
    freq = np.random.uniform(1.8, 2.5)  # Slight frequency variations
    ax = 1.2 * np.sin(2 * np.pi * freq * time_steps)
    ay = 0.5 * np.cos(2 * np.pi * freq * time_steps)
    az = np.ones(WINDOW_SIZE) * 0.8
    sample = np.stack([ax, ay, az], axis=-1) + np.random.normal(0, 0.05, size=(WINDOW_SIZE, CHANNELS))
    X_list.append(sample)
    y_list.append(1)

# Class 2: PUNCH (Forward acceleration spike followed by sudden impact stop)
for _ in range(SAMPLES_PER_CLASS):
    sample = np.random.normal(0, 0.05, size=(WINDOW_SIZE, CHANNELS))
    sample[:, 2] += 0.8  # Baseline gravity
    spike_index = np.random.randint(15, 30)  # Random strike moment
    # Sharp forward surge (+3.0g) then rebound (-2.0g)
    sample[spike_index:spike_index+3, 0] += 3.0
    sample[spike_index+3:spike_index+6, 0] -= 2.0
    X_list.append(sample)
    y_list.append(2)

x_data = np.array(X_list, dtype=np.float32)  # Shape: (1200, 50, 3)
y_data = np.array(y_list, dtype=np.int32)    # Shape: (1200,)

# Shuffle the dataset
indices = np.arange(len(x_data))
np.random.shuffle(indices)
x_data, y_data = x_data[indices], y_data[indices]

print(f"Dataset generated: X shape = {x_data.shape}, y shape = {y_data.shape}")
model=keras.Sequential(
    [
        keras.layers.Input(shape=(WINDOW_SIZE, CHANNELS)),
        keras.layers.Conv1D(filters=8, kernel_size=3,activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(3)
    ]
)

model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

model.fit(x_data,y_data,epochs=150,batch_size=32,validation_split=0.2,verbose=1)


def representative_dataset():
    for i in range(100):
        yield [x_data[i:i+1]]

def conver_to_array(array, filename, arrayname):
    hex_array = [f"0x{b:02x}" for b in array]
    hex_array = ",".join(hex_array)
    with open(f"{filename}.h", 'w') as f:
        f.write(f"const unsigned char {arrayname}[]={{ {hex_array}}};\n")
        f.write(f"const unsigned int {arrayname}_len={len(array)};\n")


converter=tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset

tf_lite_model=converter.convert()

with open("cnn.tflite","wb") as f:
    f.write(tf_lite_model)

conver_to_array(tf_lite_model, "cnn_model", "cnn_model")
print('TFLite model size:', len(tf_lite_model), 'bytes')
