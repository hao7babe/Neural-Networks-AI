class DeepNeuralNetwork:
    # Same __init__ and forward_propagation methods
    
    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = self.num_layers
        m = AL.shape[1]
        
        # Derivative of softmax + cross-entropy loss
        dZL = AL - Y
        A_prev, ZL = caches[L - 1]
        grads[f'dW{L}'] = np.dot(dZL, A_prev.T) / m
        grads[f'db{L}'] = np.sum(dZL, axis=1, keepdims=True) / m
        
        dA_prev = np.dot(self.parameters[f'W{L}'].T, dZL)
        
        # Backprop through the hidden layers
        for l in reversed(range(1, L)):
            A_prev, Z = caches[l - 1]
            dZ = dA_prev * ActivationFunctions.relu_derivative(Z)
            grads[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.num_layers + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
