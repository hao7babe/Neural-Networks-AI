# Programming Assignment:
# Start developing a Neural Network with one hidden layer
# Use object-oriented approach.
# Recommended class structure:
#        Class – Activation
#        Class – Neuron
#        Class – Layer
#        Class – Parameters
#        Class – Model (start with Neural Network with one hidden layer)
#        Class – LossFunction
#        Class – ForwardProp
#        Class – BackProp
#        Class – GradDescent
#        Class – Traning


import numpy as np
from keras.datasets import mnist
from PIL import Image

class Activation:
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, x):
        self.input = x
        self.z = np.dot(self.weights, x) + self.bias
        self.a = Activation.relu(self.z)
        return self.a

    def backward(self, delta, lr):
        dz = delta * Activation.relu_derivative(self.z)
        dw = dz * self.input
        db = dz
        self.weights -= lr * dw
        self.bias -= lr * db
        return self.weights.dot(dz)

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        if self.activation == 'softmax':
            self.a = Activation.softmax(self.z)
        elif self.activation == 'relu':
            self.a = Activation.relu(self.z)
        else:
            self.a = self.z
        return self.a

    def backward(self, delta, lr):
        if self.activation == 'softmax':
            dz = delta  
        elif self.activation == 'relu':
            dz = delta * Activation.relu_derivative(self.z)
        else:
            dz = delta

        dw = np.outer(self.input, dz)
        db = dz
        delta_prev = np.dot(self.weights, dz)

        self.weights -= lr * dw
        self.bias -= lr * db

        return delta_prev

# Loss Function Class
class LossFunction:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-9))

    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        return y_pred - y_true

class ForwardProp:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class BackProp:
    def __init__(self, layers, loss_derivative):
        self.layers = layers
        self.loss_derivative = loss_derivative

    def backward(self, y_pred, y_true, lr):
        delta = self.loss_derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr)


# Training Class
class Training:
    def __init__(self, model, loss_function, forward_prop, back_prop, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.forward_prop = forward_prop
        self.back_prop = back_prop
        self.optimizer = optimizer

    def train(self, X, y, epochs=1000, lr=0.01, print_every=100):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                y_pred = self.forward_prop.forward(X[i])
                loss = self.loss_function.cross_entropy(y_pred, y[i])
                total_loss += loss
                self.back_prop.backward(y_pred, y[i], lr)
            if epoch % print_every == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch}, Loss: {avg_loss}")

# Model Class
class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            Layer(input_size, hidden_size, activation='relu'),
            Layer(hidden_size, output_size, activation='softmax')
        ]
        self.forward_prop = ForwardProp(self.layers)
        self.back_prop = BackProp(self.layers, LossFunction.cross_entropy_derivative)
        self.loss_function = LossFunction
        self.training = Training(self, self.loss_function, self.forward_prop, self.back_prop, None)

    def predict(self, x):
        y_pred = self.forward_prop.forward(x)
        return np.argmax(y_pred)

    def train(self, X, y, epochs=1000, lr=0.01):
        self.training.train(X, y, epochs, lr)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Resize images from 28x28 to 20x20 and flatten them
    X_train_resized = np.array([
        np.array(Image.fromarray(img).resize((20, 20))).flatten()
        for img in X_train
    ])
    X_test_resized = np.array([
        np.array(Image.fromarray(img).resize((20, 20))).flatten()
        for img in X_test
    ])

    # Normalize the pixel values
    X_train_resized = X_train_resized / 255.0
    X_test_resized = X_test_resized / 255.0

    return X_train_resized, y_train, X_test_resized, y_test

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train, 10)
    y_test_one_hot = one_hot_encode(y_test, 10)

    # Initialize model
    input_size = 400  # 20x20 pixels
    hidden_size = 128  # Number of neurons in hidden layer
    output_size = 10   # Number of classes

    model = Model(input_size, hidden_size, output_size)

    # Train the model on first 1000 samples
    model.train(X_train[:1000], y_train_one_hot[:1000], epochs=1000, lr=0.01)

    # Predict on first 10 test samples
    y_pred = [model.predict(x) for x in X_test[:10]]
    y_pred_labels = y_pred

    print("Predicted labels:", y_pred_labels)
    print("True labels:", y_test[:10])

