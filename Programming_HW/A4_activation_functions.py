import numpy as np

class ActivationFunctions:
    @staticmethod
    def linear(Z):
        return Z
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def tanh(Z):
        return np.tanh(Z)
    
    @staticmethod
    def softmax(Z):
        exp_values = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)
    
    # Derivatives for backpropagation
    @staticmethod
    def linear_derivative(dA):
        return dA
    
    @staticmethod
    def relu_derivative(Z):
        return np.where(Z > 0, 1, 0)
    
    @staticmethod
    def sigmoid_derivative(Z):
        sig = ActivationFunctions.sigmoid(Z)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.power(np.tanh(Z), 2)
    
    @staticmethod
    def softmax_derivative(dA, Z):
        # Assuming dA is already calculated using cross-entropy loss
        return dA
