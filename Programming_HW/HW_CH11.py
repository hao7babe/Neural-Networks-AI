# Implement a mini batch approach.

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
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers

    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A, activation_cache = relu(Z)
        cache = (A_prev, parameters['W' + str(l)], parameters['b' + str(l)], Z)
        caches.append(cache)

    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL, activation_cache = sigmoid(ZL)
    cache = (A, parameters['W' + str(L)], parameters['b' + str(L)], ZL)
    caches.append(cache)

    return AL, caches

# Compute cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply((1 - Y), np.log(1 - AL))) / m
    return np.squeeze(cost)

# Backward propagation
def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    (A_prev, W, b, Z) = current_cache
    dZL = sigmoid_backward(dAL, Z)
    grads["dW" + str(L)] = np.dot(dZL, A_prev.T) / m
    grads["db" + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
    grads["dA" + str(L - 1)] = np.dot(W.T, dZL)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        A_prev, W, b, Z = current_cache
        dA = grads["dA" + str(l + 1)]
        dZ = relu_backward(dA, Z)
        grads["dW" + str(l + 1)] = np.dot(dZ, A_prev.T) / m
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

# Mini-batch creation
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handle the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# Model with mini-batch gradient descent
def model(X, Y, layers_dims, learning_rate=0.0075, mini_batch_size=64, num_iterations=1000):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        minibatches = random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches:
            (mini_batch_X, mini_batch_Y) = minibatch

            AL, caches = forward_propagation(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = backward_propagation(AL, mini_batch_Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    return parameters, costs

# Example usage
if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(12288, 148)  # Example input data
    Y = np.random.randint(0, 2, (1, 148))  # Binary labels

    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer neural network

    parameters, costs = model(X, Y, layers_dims, learning_rate=0.0075, mini_batch_size=32, num_iterations=1000)
