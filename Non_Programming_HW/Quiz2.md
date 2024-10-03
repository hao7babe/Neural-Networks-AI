Here's a description of the most commonly used activation functions in neural networks:

1. **Linear Activation Function**:
   - **Equation**: \( f(x) = x \)
   - **Description**: The output is directly proportional to the input, making it essentially a straight line. While simple, it's not commonly used in hidden layers of neural networks because it doesn't introduce non-linearity. Without non-linearity, the network cannot learn complex patterns.
   - **Usage**: Often used in the output layer of a regression model, where the task is to predict continuous values.

2. **ReLU (Rectified Linear Unit)**:
   - **Equation**: \( f(x) = \max(0, x) \)
   - **Description**: ReLU outputs the input directly if it's positive, and 0 otherwise. It introduces non-linearity, making it one of the most widely used activation functions in deep learning models.
   - **Usage**: Commonly used in hidden layers of deep neural networks due to its efficiency in computation and ability to help with vanishing gradient problems. ReLU accelerates convergence during training.

3. **Sigmoid Activation Function**:
   - **Equation**: \( f(x) = \frac{1}{1 + e^{-x}} \)
   - **Description**: The sigmoid function squashes the input to the range (0, 1). It's often interpreted as a probability in classification tasks. However, the gradient for large or small values tends to be close to zero, which can slow down training (vanishing gradient problem).
   - **Usage**: Typically used in the output layer for binary classification problems. Sometimes used in simpler models, but less common in deeper networks due to its limitations.

4. **Tanh (Hyperbolic Tangent) Activation Function**:
   - **Equation**: \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
   - **Description**: Similar to the sigmoid function, but the output is scaled between -1 and 1. Tanh is zero-centered, making it preferred over sigmoid in some cases, but still suffers from vanishing gradients for very large or small values of input.
   - **Usage**: Commonly used in hidden layers, especially in networks that need to learn complex relationships, such as recurrent neural networks (RNNs). It’s favored when the input data is centered around zero.

5. **Softmax Activation Function**:
   - **Equation**: \( f(x_i) = \frac{e^{x_i}}{\sum_{j}e^{x_j}} \)
   - **Description**: The softmax function outputs a probability distribution, where the sum of all outputs is equal to 1. Each output value is normalized between 0 and 1 based on the input.
   - **Usage**: Used in the output layer of neural networks for multi-class classification problems. It assigns a probability to each class, making it suitable for predicting categorical labels.

These activation functions play different roles depending on the task (e.g., classification vs. regression) and layer types (e.g., hidden layers vs. output layers) within a neural network.