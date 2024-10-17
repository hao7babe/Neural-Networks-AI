### 1. Describe the artificial neuron model (4 points)
An **artificial neuron** is the basic unit of a neural network, similar to a biological neuron. It takes several inputs, applies weights to them, adds a bias, and then passes the result through an activation function to produce an output.

- **Formula**:  
  \( y = f(w_1x_1 + w_2x_2 + ... + w_nx_n + b) \), where:
  - \( w \) are the weights for each input
  - \( x \) are the input values
  - \( b \) is the bias
  - \( f \) is the activation function

- **Example**:  
  Imagine a neuron receiving two inputs: \( x_1 = 0.5 \) and \( x_2 = 0.8 \), with weights \( w_1 = 0.6 \) and \( w_2 = 0.4 \), plus a bias \( b = 0.1 \). The weighted sum is:  
  \( 0.5 \times 0.6 + 0.8 \times 0.4 + 0.1 = 0.58 \).  
  This value would then go through an activation function (like the sigmoid function) to produce the final output.

### 2. What is the logistic regression problem? (4 points)
**Logistic regression** is a type of algorithm used for binary classification problems. It predicts the probability of an event happening by using a sigmoid function to produce an output between 0 and 1.

- **Formula**:  
  \( P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}} \), where:
  - \( w \) is the vector of weights
  - \( x \) are the input features
  - \( b \) is the bias term
  
- **Example**:  
  Logistic regression might predict whether an email is spam or not based on features like the presence of certain keywords, the sender, etc. It outputs a probability between 0 (not spam) and 1 (spam), and a threshold (like 0.5) is applied to classify it.

### 3. Describe multilayer (deep) neural network (4 points)
A **multilayer (deep) neural network** has multiple layers of neurons: an input layer, one or more hidden layers, and an output layer. Each layer transforms its input through weighted connections and activation functions, allowing the network to learn complex patterns and relationships.

- **Example**:  
  A deep neural network used to classify images might learn low-level features like edges in the first hidden layer and more complex shapes or patterns in deeper layers. This enables the network to classify images, like distinguishing between cats and dogs.

### 4. Describe major activation functions and explain their usage (4 points)
- **Linear**: The output is the same as the input. This function is mainly used in regression tasks but is not useful for hidden layers in deep learning.
  - **Formula**: \( f(x) = x \)
  - **Example**: In a network predicting house prices, a linear activation can be used in the output layer to predict continuous values.

- **ReLU (Rectified Linear Unit)**: Outputs the input if it’s positive, and 0 otherwise. It’s commonly used in hidden layers of deep neural networks due to its efficiency in training.
  - **Formula**: \( f(x) = \max(0, x) \)
  - **Example**: ReLU is often used in deep networks for image recognition because it helps models converge faster.

- **Sigmoid**: Converts input values to a range between 0 and 1. It’s mainly used for binary classification problems.
  - **Formula**: \( f(x) = \frac{1}{1 + e^{-x}} \)
  - **Example**: Sigmoid is used in logistic regression to predict the probability that an email is spam or not.

- **Tanh**: Similar to the sigmoid function but ranges from -1 to 1, which helps center the data. It’s commonly used in hidden layers of neural networks.
  - **Formula**: \( f(x) = \tanh(x) \)
  - **Example**: Tanh is useful in cases where negative values are meaningful, such as predicting temperature changes over time.

- **Softmax**: Turns a vector of numbers into probabilities that sum to 1. It’s typically used in the output layer for multi-class classification.
  - **Formula**: \( f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}} \)
  - **Example**: Softmax is used in neural networks that classify images into multiple categories, such as identifying whether a picture is a cat, dog, or bird.

### 5. What is supervised learning? (4 points)
**Supervised learning** is a machine learning approach where the model is trained on labeled data. Each input has a corresponding correct output (label), and the model learns to map inputs to outputs by minimizing the error in its predictions.

- **Example**:  
  Predicting house prices based on historical data where features like square footage and number of bedrooms are inputs, and the sale price is the label.

### 6. Describe loss/cost function (4 points)
A **loss function** measures how well a neural network's predictions match the actual data. It computes the error for each prediction, and the goal of training is to minimize this error.

- **Example**:  
  For binary classification tasks, **cross-entropy loss** is commonly used. It penalizes incorrect predictions more heavily when the model is very confident but wrong.  
  \[
  L = - \frac{1}{N} \sum (y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i))
  \]
  Where \( y_i \) is the actual label and \( \hat{y}_i \) is the predicted probability.

### 7. Describe forward and backward propagation for a multilayer (deep) neural network (4 points)
- **Forward propagation**: In this phase, the input data is passed through the network layer by layer. Each layer processes the input, applies the activation function, and sends the result to the next layer. This continues until the final output is produced, which is compared with the actual output to calculate the loss.
  
- **Backward propagation**: After calculating the loss, the error is propagated back through the network. Gradients are calculated for each layer using the chain rule, and these gradients are used to update the weights and biases, reducing the error for the next round of training.

- **Example**:  
  In an image classification network, forward propagation generates a prediction (like cat or dog), and backward propagation adjusts the network’s weights based on how far the prediction was from the true label.

### 8. What are parameters and hyperparameters in neural networks, and what is the conceptual difference between them? (4 points)
- **Parameters**: These are the values that the model learns during training, such as the weights and biases.
  
- **Hyperparameters**: These are settings that control the training process but aren’t learned from the data. They include things like learning rate, batch size, and the number of layers in the network.

- **Difference**: Parameters are updated during training based on the data, while hyperparameters are set before training starts and control how the model learns.

- **Example**:  
  The connections between neurons in a network have weights, which are parameters. The learning rate, which controls how fast the model updates its weights, is a hyperparameter.

### 9. How to set the initial values for the neural network training (4 points)
Initializing weights is important to ensure the network trains properly. Two common strategies are:
- **Xavier Initialization**: Used when the network uses sigmoid or tanh activations. It helps prevent the gradients from becoming too small or too large.
  - **Formula**: \( W \sim U\left[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}\right] \), where \( n \) is the number of inputs to a neuron.

- **He Initialization**: Often used with ReLU activation functions to prevent exploding or vanishing gradients.
  - **Formula**: \( W \sim N\left(0, \frac{2}{n}\right) \), where \( n \) is the number of inputs to a neuron.

- **Example**:  
  Xavier initialization is commonly used for deep neural networks with sigmoid or tanh activations to keep the gradient values reasonable during training.

### 10. Why are mini-batches used instead of complete batches in training of neural networks? (4 points)
**Mini-batches** balance the advantages of using the entire dataset (batch gradient descent) and using individual examples (stochastic gradient descent). They allow for faster training and better use of computational resources, while also smoothing out the learning process by averaging over multiple examples.

- **Example**:  
  Instead of using all 10,000 examples in a dataset at once (which is computationally expensive), a mini-batch of 32 samples is processed. This allows for quicker updates to the model weights while still capturing the general trend of the data.