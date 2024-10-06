import numpy as np

class Softmax:
    def __init__(self):
        self.cache = None

    def forward(self, Z):
        """
        Implements the forward propagation for softmax activation.
        
        Args:
        Z -- The linear component (Z) of the output layer, a numpy array of shape (n_classes, m), 
             where n_classes is the number of output classes and m is the number of examples.
        
        Returns:
        A -- The activation value using softmax, numpy array of shape (n_classes, m)
        """
        # Subtracting the max value from Z for numerical stability
        Z_stable = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_stable)
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        self.cache = A
        return A

    def backward(self, Y):
        """
        Implements the backward propagation for softmax activation.
        
        Args:
        Y -- True labels in one-hot encoded form, numpy array of shape (n_classes, m)
        
        Returns:
        dZ -- The gradient of the loss with respect to Z, numpy array of shape (n_classes, m)
        """
        m = Y.shape[1]
        dZ = self.cache - Y
        return dZ / m

# Example usage of the Softmax class
if __name__ == "__main__":
    np.random.seed(1)
    
    # Example input (Z) and true labels (Y)
    Z = np.random.randn(3, 5)  # 3 classes, 5 examples
    Y = np.array([[1, 0, 0, 0, 1], 
                  [0, 1, 0, 1, 0], 
                  [0, 0, 1, 0, 0]])  # One-hot encoded labels

    # Initialize softmax class
    softmax = Softmax()

    # Forward propagation
    A = softmax.forward(Z)
    print("Softmax Activations:")
    print(A)

    # Backward propagation
    dZ = softmax.backward(Y)
    print("\nGradients of the Loss w.r.t. Z:")
    print(dZ)
