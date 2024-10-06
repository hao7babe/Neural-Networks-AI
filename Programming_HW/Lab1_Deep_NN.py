import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# Task 1: Define the Neural Network Model
model = Sequential([
    Dense(10, activation='relu', input_dim=10),  # Layer 1: 10 neurons, ReLU
    Dense(8, activation='relu'),  # Layer 2: 8 neurons, ReLU
    Dense(8, activation='relu'),  # Layer 3: 8 neurons, ReLU
    Dense(4, activation='relu'),  # Layer 4: 4 neurons, ReLU
    Dense(1, activation='sigmoid')  # Layer 5 (Output): 1 neuron, Sigmoid
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Task 2: Generate Synthetic Training Data
np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(0, 2, size=(1000, 1))  # Binary labels (0 or 1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Print model summary (optional)
model.summary()
