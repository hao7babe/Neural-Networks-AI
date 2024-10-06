### 1. What is normalization and why is it needed?

- **Normalization**: Normalization is a technique used to scale input features to a consistent range, typically between 0 and 1, or -1 and 1. This process ensures that all features contribute equally to the learning process, regardless of their original scale.

- **Why it's needed**:
  - **Improves convergence**: Normalized data helps gradient-based optimization algorithms, like gradient descent, converge faster by maintaining stable gradients.
  - **Prevents dominance of large-scale features**: Without normalization, features with larger values may dominate the learning process, leading to poor model performance.
  - **Enhances model accuracy**: In models like neural networks or k-nearest neighbors, normalization ensures that all features are treated equally during distance or weight calculations.

### 2. What are vanishing and exploding gradients?

- **Vanishing Gradients**: This occurs when the gradients (partial derivatives) used to update weights in deep networks become very small as they propagate back through the layers, especially in networks with many layers. As a result, the weights in earlier layers update very slowly, leading to poor training performance or even the inability to learn.

- **Exploding Gradients**: This is the opposite problem, where gradients become excessively large as they propagate back through the network. This leads to unstable updates of the model weights, which can cause the model to diverge or result in extremely large weights, making training impossible.

Both issues are more common in deep neural networks and can significantly hinder model training.

### 3. What is the Adam algorithm and why is it needed?

- **Adam Algorithm**: Adam (short for Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two other methods: momentum and RMSProp. It computes adaptive learning rates for each parameter by tracking both the first moment (mean) and the second moment (variance) of the gradients during training.

- **Why it’s needed**:
  - **Efficient and fast**: Adam adapts learning rates for each parameter, making it faster and more efficient than traditional gradient descent methods.
  - **Less tuning required**: It typically works well out of the box with default hyperparameters and requires less tuning than other optimizers.
  - **Handles noisy data well**: Adam is robust when dealing with sparse gradients and noisy data, which makes it widely used in deep learning applications.

### 4. How to choose hyperparameters?

Choosing hyperparameters is often a trial-and-error process, but here are some general guidelines:

- **Learning Rate**: Start with a small value, like 0.001, and adjust based on how quickly or slowly the model converges. Too large a learning rate might cause divergence, while too small a rate may result in slow convergence.
  
- **Batch Size**: Common batch sizes are 32, 64, or 128. Smaller batch sizes can give noisy updates but may lead to better generalization. Larger batch sizes offer smoother updates but require more memory.
  
- **Number of Epochs**: Choose enough epochs to ensure the model has learned well without overfitting. Early stopping based on validation performance can help prevent overfitting.
  
- **Regularization**: L1, L2 regularization, or dropout can be tuned by adjusting their strength (penalty factor) to prevent overfitting. Too much regularization can cause underfitting.
  
- **Optimizer**: If using Adam, its default parameters (learning rate = 0.001, \( \beta_1 \)=0.9, \( \beta_2 \)=0.999) work well for most problems, but fine-tuning them might be necessary for specific tasks.

- **Grid Search / Random Search**: Techniques like grid search or random search can be used to automate hyperparameter tuning by testing a range of values and selecting the best combination based on validation performance.