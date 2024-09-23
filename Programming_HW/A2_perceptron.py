# Programming Assignment:
# Prepare a set of grayscale images of handwritten labeled numbers 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 as input for training neural networks. The size of the images is 20 x 20. Make 10 different images of reach number written a little differently.
# Prepare some test images (not labeled) for testing the trained Perceptron.
# Develop the software for the Perceptron with input, parameters as transmission matrix and bias including the module for their initiation (setting their initial values), sigmoid activation function, loss function generation, forward and backpropagation paths, gradient descent algorithm for training. Make sure that the software allows easy change of set of images, activation function (various activation functions may be used in the future),
# Initiate the parameters (transmission matrix and bias).
# Try to train and test the Perceptron.
# Do not get disappointed if you face hardship in training the Perceptron. Try to resolve the problems, consult with the TA, and collect the unresolved problems for a discussion in the next class.

import numpy as np
from keras.datasets import mnist
from PIL import Image

class Perceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes)  
        self.bias = np.random.randn(num_classes)  

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z)) 
        return exp_z / np.sum(exp_z)

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias 
        return self.softmax(z)

    def loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-9))  

    def backward(self, x, y_true, y_pred, lr):
        dz = y_pred - y_true
        dw = np.outer(x, dz)  
        db = dz  
        self.weights -= lr * dw
        self.bias -= lr * db

    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                y_pred = self.forward(X[i])
                total_loss += self.loss(y_pred, y[i])
                self.backward(X[i], y[i], y_pred, lr)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Resize images from 28x28 to 20x20 and flatten them
    X_train_resized = np.array([np.array(Image.fromarray(img).resize((20, 20))).flatten() for img in X_train])
    X_test_resized = np.array([np.array(Image.fromarray(img).resize((20, 20))).flatten() for img in X_test])

    # Normalize the pixel values
    X_train_resized = X_train_resized / 255.0
    X_test_resized = X_test_resized / 255.0

    return X_train_resized, y_train, X_test_resized, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_data()

    y_train_one_hot = one_hot_encode(y_train, 10)
    y_test_one_hot = one_hot_encode(y_test, 10)

    model = Perceptron(input_size=400, num_classes=10)

    model.train(X_train[:1000], y_train_one_hot[:1000], epochs=1000, lr=0.01)

    y_pred = [model.forward(x) for x in X_test[:10]] 
    y_pred_labels = [np.argmax(pred) for pred in y_pred]

    print("Predicted labels:", y_pred_labels)
    print("True labels:", y_test[:10])
