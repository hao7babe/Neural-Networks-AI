### 1. What is learning rate decay and why is it needed?

- **Learning Rate Decay**: Learning rate decay is the technique of gradually reducing the learning rate as training progresses. Initially, a higher learning rate allows the model to learn quickly, but as the model approaches an optimal solution, a smaller learning rate helps fine-tune the model and avoid overshooting the optimal point.

- **Why it's needed**:
  - **Improves convergence**: By lowering the learning rate over time, the model can settle into a more accurate solution.
  - **Prevents overshooting**: In later stages of training, a smaller learning rate ensures that updates to weights are fine-tuned, preventing the model from overshooting the optimal point.

### 2. What are saddle and plateau problems?

- **Saddle Points**: These are points in the loss function where the gradient is zero but the point is not a local minimum. The gradient descent algorithm might get stuck at saddle points, slowing down the learning process as it struggles to find a proper direction to move in.

- **Plateau Problems**: A plateau is a flat region in the loss function where the gradient is close to zero over a large area, making progress slow. The optimizer can spend a lot of time stuck in these regions, leading to slower training.

Both problems hinder the learning process, as the optimizer struggles to make progress.

### 3. Why should we avoid the grid approach in hyperparameter choice?

- **Grid Search**: In grid search, we try all possible combinations of hyperparameter values within a predefined grid of values. While this can give an exhaustive search of the parameter space, it's inefficient for the following reasons:
  - **Expensive and Time-Consuming**: Grid search requires testing every combination, which becomes computationally expensive as the number of hyperparameters increases (combinatorial explosion).
  - **Inflexible**: It does not account for the importance of different hyperparameters, treating them all equally, which may lead to unnecessary exploration in some areas of the hyperparameter space.
  
  **Alternatives**: Random search and Bayesian optimization are often more efficient approaches, as they focus the search on more promising regions of the hyperparameter space and reduce unnecessary computations.

### 4. What is a mini-batch and how is it used?

- **Mini-Batch**: A mini-batch is a small subset of the training data used to calculate the gradient and update the model's weights during each iteration of training. Instead of using the entire dataset (as in batch gradient descent) or a single sample (as in stochastic gradient descent), mini-batch gradient descent strikes a balance by processing smaller chunks of the dataset.

- **How it's used**:
  - The model updates its weights based on the gradients computed from the mini-batch, which allows for more frequent updates than full-batch gradient descent.
  - Typical mini-batch sizes range from 32 to 512 samples, depending on memory and computational power.
  - **Advantages**: It speeds up training, reduces memory usage, and introduces some noise in updates, which can help the model generalize better by avoiding overfitting.

In summary, mini-batch gradient descent combines the advantages of both batch and stochastic methods, providing faster and more efficient training without sacrificing too much stability.