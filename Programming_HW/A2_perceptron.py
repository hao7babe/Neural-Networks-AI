# Programming Assignment:
# Prepare a set of grayscale images of handwritten labeled numbers 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 as input for training neural networks. The size of the images is 20 x 20. Make 10 different images of reach number written a little differently.
# Prepare some test images (not labeled) for testing the trained Perceptron.
# Develop the software for the Perceptron with input, parameters as transmission matrix and bias including the module for their initiation (setting their initial values), sigmoid activation function, loss function generation, forward and backpropagation paths, gradient descent algorithm for training. Make sure that the software allows easy change of set of images, activation function (various activation functions may be used in the future),
# Initiate the parameters (transmission matrix and bias).
# Try to train and test the Perceptron.
# Do not get disappointed if you face hardship in training the Perceptron. Try to resolve the problems, consult with the TA, and collect the unresolved problems for a discussion in the next class.

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random

# PART 1: Generate Grayscale Digit Images (20x20)
def generate_digit_images():
    digits = list(range(10))
    for digit in digits:
        for i in range(10):
            img = Image.new('L', (20, 20), color=255)  # 'L' mode for grayscale
            draw = ImageDraw.Draw(img)

            # Draw the digit in the center with slight randomness
            x_offset = random.randint(1, 5)
            y_offset = random.randint(1, 5)
            font_size = random.randint(12, 16)
            draw.text((x_offset, y_offset), str(digit), fill=0)

            img.save(f'digits/digit_{digit}_{i}.png')

# PART 2: Generate Test Images (Without Labels)
def generate_test_images():
    for i in range(20):  # Generate 20 test images
        img = Image.new('L', (20, 20), color=255)
        draw = ImageDraw.Draw(img)

        digit = random.choice(list(range(10)))
        x_offset = random.randint(1, 5)
        y_offset = random.randint(1, 5)
        font_size = random.randint(12, 16)
        draw.text((x_offset, y_offset), str(digit), fill=0)

        img.save(f'test_digits/test_image_{i}.png')

# PART 3: Activation Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# PART 4: Loss Function (Mean Squared Error)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# PART 5: Initialize Weights and Biases
def initialize_parameters(input_size, output_size):
    W = np.random.randn(output_size, input_size) * 0.01  # Transmission matrix (weights)
    b = np.zeros((output_size, 1))  # Bias
    return W, b

# PART 6: Forward Propagation
def forward_propagation(X, W, b):
    Z = np.dot(W, X) + b  # Weighted sum
    A = sigmoid(Z)  # Activation function
    return A, Z

# PART 7: Backward Propagation
def backward_propagation(X, Y, A, W, b, Z, learning_rate):
    m = X.shape[1]  # Number of samples
    dZ = A - Y  # Derivative of loss w.r.t Z
    dW = np.dot(dZ, X.T) / m  # Gradient of weights
    db = np.sum(dZ, axis=1, keepdims=True) / m  # Gradient of bias
    
    # Update weights and biases
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

# PART 8: Training the Perceptron
def train_perceptron(X_train, Y_train, input_size, output_size, learning_rate=0.01, epochs=1000):
    W, b = initialize_parameters(input_size, output_size)
    
    for epoch in range(epochs):
        # Forward propagation
        A, Z = forward_propagation(X_train, W, b)
        
        # Compute loss
        loss = mse_loss(Y_train, A)
        
        # Backward propagation and update
        W, b = backward_propagation(X_train, Y_train, A, W, b, Z, learning_rate)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return W, b

# PART 9: Prediction Function
def predict(X_test, W, b):
    A, _ = forward_propagation(X_test, W, b)
    predictions = A > 0.5  # Binary classification
    return predictions

# PART 10: Load Images and Preprocess
def load_images(folder_path):
    images = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            label = int(file_name.split('_')[1])  # Extract label from filename
            img = Image.open(os.path.join(folder_path, file_name)).convert('L')  # Convert to grayscale
            img = np.asarray(img).reshape(-1, 1) / 255.0  # Normalize and flatten
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# PART 11: Main Function
if __name__ == "__main__":
    # Step 1: Generate the digit images for training and testing
    generate_digit_images()
    generate_test_images()

    # Step 2: Load training data
    X_train, Y_train = load_images('digits')  # You need to provide the path to the 'digits' folder

    # Step 3: Train the perceptron
    input_size = 400  # 20x20 images
    output_size = 1  # For binary classification (digit recognition will need more outputs later)
    
    W, b = train_perceptron(X_train, Y_train, input_size, output_size)
    
    # Step 4: Test the perceptron on test data
    X_test, _ = load_images('test_digits')  # Load test images
    predictions = predict(X_test, W, b)
    print(f'Predictions: {predictions}')
