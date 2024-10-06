# Develop a prototype of the training, validation, and testing sets of your choice for the future training of your neural network. The term “prototype: means that the sets may be quite limited by number of images, but the proportions of images in them should be maintained as required

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# Load CIFAR-10 dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values (optional but recommended for neural networks)
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split the training data into training and validation sets (85% training, 15% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.15, random_state=42)

# Confirm the sizes of each set
print(f"Full Training set size: {x_train.shape[0]}")
print(f"Validation set size: {x_val.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")

# Optional: For a smaller prototype, let's take 1000 samples from the full dataset
sample_size = 1000
x_sampled, _, y
