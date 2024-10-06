# Programming Assignment:
# Implement regularization algorithms in your neural network.
# Implement dropout algorithms in your neural network.

import numpy as np

# Helper functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z

def relu(Z):
    return np.maximum(0, Z), Z

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# Forward propagation with dropout
def forward_propagation(X, parameters, keep_prob=1):
    np.random.seed(1)
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers

    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A, activation_cache = relu(Z)
        cache = (A_prev, parameters['W' + str(l)], parameters['b' + str(l)], Z)
        caches.append(cache)

        # Implementing dropout
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A = A * D
        A = A / keep_prob
        cache = (cache, D)
        caches[-1] = cache

    # Output layer
    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL, activation_cache = sigmoid(ZL)
    cache = (A, parameters['W' + str(L)], parameters['b' + str(L)], ZL)
    caches.append(cache)

    return AL, caches

# Compute cost with L2 regularization
def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    cross_entropy_cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply((1 - Y), np.log(1 - AL))) / m

    L = len(parameters) // 2
    L2_regularization_cost = 0
    for l in range(1, L + 1):
        L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))

    L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost

    cost = cross_entropy_cost + L2_regularization_cost
    return cost

# Backward propagation with dropout and L2 regularization
def backward_propagation(AL, Y, caches, lambd, keep_prob):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Output layer gradient
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    (A_prev, W, b, Z) = current_cache
    dZL = sigmoid_backward(dAL, Z)
    grads["dW" + str(L)] = np.dot(dZL, A_prev.T) / m + (lambd / m) * W
    grads["db" + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
    grads["dA" + str(L - 1)] = np.dot(W.T, dZL)

    # Hidden layer gradients
    for l in reversed(range(L - 1)):
        current_cache, D = caches[l]
        A_prev, W, b, Z = current_cache
        dA = grads["dA" + str(l + 1)]
        dA = dA * D  # Apply dropout mask
        dA = dA / keep_prob
        dZ = relu_backward(dA, Z)
        grads["dW" + str(l + 1)] = np.dot(dZ, A_prev.T) / m + (lambd / m) * W
        grads["db" + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True) / m
        grads["dA" + str(l)] = np.dot(W.T, dZ)

    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters

# Model function
def model(X, Y, layers_dims, learning_rate=0.0075, lambd=0, keep_prob=1, num_iterations=3000):
    parameters = initialize_parameters(layers_dims)
    costs = []

    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters, keep_prob)
        cost = compute_cost_with_regularization(AL, Y, parameters, lambd)
        grads = backward_propagation(AL, Y, caches, lambd, keep_prob)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    return parameters, costs

# Example usage:
if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(12288, 10)  # 10 samples, 12288 features (for example, 64x64 image flattened)
    Y = np.random.randint(0, 2, (1, 10))  # Binary classification output for 10 samples
    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

    parameters, costs = model(X, Y, layers_dims, learning_rate=0.0075, lambd=0.7, keep_prob=0.8, num_iterations=2500)

