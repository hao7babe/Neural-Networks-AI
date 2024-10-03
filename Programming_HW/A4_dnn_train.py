class DeepNeuralNetwork:
    # Same __init__, forward_propagation, backward_propagation, update_parameters methods
    
    def train(self, X, Y, epochs, learning_rate):
        for i in range(epochs):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_loss(AL, Y)
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)
            
            if i % 100 == 0:
                print(f'Cost after epoch {i}: {cost}')

