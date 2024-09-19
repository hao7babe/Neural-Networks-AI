## HW to Chapter 3 & 4 “The Perceptron for Logistic Regression“

1. **Describe the logistic regression**:
    - Logistic regression is used for binary classification, predicting two possible outcomes (e.g., 0 or 1, yes or no).
    - The model calculates a weighted sum of input features and passes the result through the sigmoid function to output a probability between 0 and 1.
    - The decision threshold (commonly 0.5) determines which class the input belongs to.
2. **How are grayscale and color (RGB) images presented as inputs for the perceptron?**
    - **Grayscale images** are represented as a 2D matrix where each pixel value (ranging from 0 to 255) indicates the intensity of light. 0 corresponds to black, and 255 to white.
    - **RGB images** are represented as a 3D array where each pixel contains three values representing the Red, Green, and Blue channels. Each value in these channels also ranges from 0 to 255. Combining these values produces various colors.
    - For a perceptron, these pixel values are often flattened into a single vector (1D array) before being fed into the model as inputs.
3. **Is image recognition a logistic regression problem? Why?**
    - **No**, image recognition typically involves recognizing many different objects or patterns within images, which is a multi-class classification problem, not a binary classification problem.
    - Logistic regression is only suitable for binary classification. Image recognition tasks typically require more complex models like convolutional neural networks (CNNs), which can capture spatial relationships between pixels.
4. **Is home prices prediction a logistic regression problem? Why?**
    - **No**, predicting home prices is a **regression problem** where the goal is to predict a continuous value (the price).
    - Logistic regression is used for classification tasks, not for predicting continuous outputs.
5. **Is image diagnostics a logistic regression problem? Why?**
    - **Yes**, if the goal is to classify an image into one of two categories, such as diagnosing an image as healthy or unhealthy.
    - Logistic regression is appropriate when you have a binary classification problem, such as disease diagnosis, where the model predicts the probability of belonging to one of the two classes.
6. **How does gradient descent optimization work?**
    - Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the model’s weights.
    - The model starts with random weights, and for each step, it calculates the gradient (the direction in which the error increases) of the loss function with respect to the weights.
    - The weights are updated in the opposite direction of the gradient by a small amount (learning rate) to reduce the loss. This process repeats until the loss converges to a minimum.
7. **How does image recognition work as logistic regression classifier?**
    - In binary image recognition tasks (e.g., cat vs. no cat), the pixel values of the image are used as input features for logistic regression.
    - Logistic regression computes a weighted sum of the pixel values, applies the sigmoid function, and outputs a probability indicating which class the image belongs to.
    - However, logistic regression is limited for image recognition due to its inability to capture complex spatial features, which is why deep learning models like CNNs are preferred.
8. **Describe the logistic regression loss function and explain the reasons behind this choice.**
    - The loss function used in logistic regression is **Binary Cross-Entropy Loss**.
    - This function measures the difference between the predicted probability and the true binary label (0 or 1).
    - It is chosen because it is convex, meaning it has a single global minimum, which makes it easier to optimize using gradient descent.
    - The loss function penalizes incorrect predictions more severely, guiding the model to adjust its weights and improve accuracy.
9. **Describe the sigmoid activation function and the reasons behind its choice.**
    - The sigmoid function is used in logistic regression to map the output of the linear equation (weighted sum of inputs) to a probability between 0 and 1.
    - Mathematically, it is defined as: $\sigma(x) = \frac{1}{1 + e^{-x}}$
    - The sigmoid function is chosen because it outputs values between 0 and 1, making it suitable for binary classification tasks where the output needs to represent a probability.
    - Additionally, it is differentiable, allowing gradient-based optimization methods like gradient descent to be applied effectively.