class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        self.parameters = self.initialize_parameters(layer_dims)
        self.num_layers = len(layer_dims) - 1
    
    def initialize_parameters(self, layer_dims):
        parameters = {}
        np.random.seed(1)  # Ensure reproducibility
        for l in range(1, len(layer_dims)):
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        return parameters
    
    def forward_propagation(self, X):
        caches = []
        A = X
        L = self.num_layers
        
        for l in range(1, L):
            A_prev = A
            Z = np.dot(self.parameters[f'W{l}'], A_prev) + self.parameters[f'b{l}']
            A = ActivationFunctions.relu(Z)
            caches.append((A_prev, Z))
        
        # Last layer using softmax
        ZL = np.dot(self.parameters[f'W{L}'], A) + self.parameters[f'b{L}']
        AL = ActivationFunctions.softmax(ZL)
        caches.append((A, ZL))
        
        return AL, caches
    
    def compute_loss(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL)) / m
        return np.squeeze(cost)  # Ensure the cost is a scalar
