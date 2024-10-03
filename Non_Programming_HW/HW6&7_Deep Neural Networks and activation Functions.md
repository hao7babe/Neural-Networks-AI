### Why are multilayer (deep) neural networks needed?
Multilayer (deep) neural networks are essential because they can model complex, non-linear relationships. Shallow networks (with fewer layers) struggle with capturing intricate data patterns, whereas deep networks with multiple hidden layers can learn hierarchical features. Each layer learns more abstract features from the raw input, making it suitable for complex tasks like image recognition, natural language processing, and more.

### What is the structure of the weight matrix (how many rows and columns)?
The structure of the weight matrix depends on the layer it connects:
- **Rows**: Equal to the number of neurons in the current layer.
- **Columns**: Equal to the number of neurons in the previous layer.

For example, if layer \( L-1 \) has 10 neurons and layer \( L \) has 5 neurons, the weight matrix will be of size \( 5 \times 10 \).

### Describe the gradient descent method
Gradient descent is an optimization method used to minimize the loss function by iteratively updating the model's weights. The process involves:
1. **Initialize weights**: Start with random or small values for weights.
2. **Compute gradient**: Calculate the gradient of the loss function with respect to each weight (i.e., partial derivatives).
3. **Update weights**: Adjust weights by moving in the opposite direction of the gradient, scaled by a learning rate.

Weight update rule:
\[
W := W - \alpha \cdot \nabla L(W)
\]
where \( W \) is the weight, \( \alpha \) is the learning rate, and \( \nabla L(W) \) is the gradient of the loss function \( L \).

### Describe in detail forward propagation and backpropagation for deep neural networks

#### Forward Propagation
Forward propagation is the process of passing inputs through the network to compute the output:
1. **Input Layer**: Inputs are fed into the input layer.
2. **Linear Transformation**: At each layer, the input \( X \) is multiplied by the weight matrix \( W \), and the bias \( b \) is added.
   \[
   Z^{[l]} = W^{[l]}X^{[l-1]} + b^{[l]}
   \]
3. **Activation Function**: The linear output \( Z \) is passed through an activation function to introduce non-linearity.
   \[
   A^{[l]} = \text{activation}(Z^{[l]})
   \]
4. **Output Layer**: The final output layer provides predictions.

#### Backpropagation
Backpropagation is the process of calculating gradients and updating weights:
1. **Calculate Loss**: Compute the difference between predicted output and actual target using a loss function (e.g., mean squared error).
2. **Compute Gradients**: Start from the output layer and calculate the gradient of the loss with respect to each weight using the chain rule.
3. **Update Weights**: Use the gradients to update the weights via gradient descent.

### Describe linear, ReLU, sigmoid, tanh, and softmax activation functions

#### Linear Activation
- **Formula**: \( f(x) = x \)
- **Purpose**: Used when no non-linearity is needed (e.g., regression tasks).
- **Typical Use**: Output layer of a regression model.

#### ReLU (Rectified Linear Unit)
- **Formula**: \( f(x) = \max(0, x) \)
- **Purpose**: Introduces non-linearity and avoids vanishing gradient by allowing only positive values.
- **Typical Use**: Hidden layers in deep networks.

#### Sigmoid
- **Formula**: \( f(x) = \frac{1}{1 + e^{-x}} \)
- **Purpose**: Squashes input values between 0 and 1, representing probabilities.
- **Typical Use**: Output layer for binary classification tasks.

#### Tanh (Hyperbolic Tangent)
- **Formula**: \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
- **Purpose**: Maps inputs between -1 and 1, centered at zero for better optimization.
- **Typical Use**: Hidden layers for tasks where negative values are needed.

#### Softmax
- **Formula**: \( f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \)
- **Purpose**: Converts logits into a probability distribution across multiple classes.
- **Typical Use**: Output layer for multi-class classification.