import numpy as np

# Function to normalize input data
def normalize_input(X):
    """
    Normalizes the input features by subtracting the mean and dividing by the standard deviation.
    
    Args:
    X -- Input data, numpy array of shape (number of features, number of examples)
    
    Returns:
    X_norm -- Normalized input data
    mean -- The mean of each feature (for later use if needed)
    std -- The standard deviation of each feature (for later use if needed)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    
    X_norm = (X - mean) / std
    return X_norm, mean, std

# Example usage
if __name__ == "__main__":
    np.random.seed(1)
    # Example input data: 4 features and 5 examples
    X = np.random.randn(4, 5)

    print("Original Input Data:")
    print(X)
    
    # Normalize the input data
    X_norm, mean, std = normalize_input(X)
    
    print("\nNormalized Input Data:")
    print(X_norm)
    
    print("\nMean of each feature (column):")
    print(mean)
    
    print("\nStandard deviation of each feature (column):")
    print(std)
