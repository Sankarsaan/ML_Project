import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

# 1. Load Data (The famous MNIST dataset)

print("Loading data...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Preprocessing (Normalization)
# Pixel values are 0-255. We scale them to 0-1 for faster convergence (Gradient Descent)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Build the Model (Neural Network)
# This matches the architecture from Course 2: Input -> Hidden (ReLU) -> Output (Softmax/Linear)
model = Sequential([
    Flatten(input_shape=(28, 28)),      # Input Layer: Flattens 28x28 image to 784 pixels
    Dense(128, activation='relu'),      # Hidden Layer 1: 128 neurons, ReLU activation
    Dense(128, activation='relu'),      # Hidden Layer 2: 128 neurons, ReLU activation
    Dense(10, activation='linear')      # Output Layer: 10 units (for digits 0-9)
])

# 4. Compile the Model
# We use 'from_logits=True' because it's more numerically stable 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 5. Train the Model
print("Training model... (This usually takes 1-2 minutes)")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# 7
model.save('handwritten_digit_model.h5')

print("Model saved as 'handwritten_digit_model.h5'")
