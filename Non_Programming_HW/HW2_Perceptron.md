# HW to Chapters 2 “The Perceptron”

## 1. Describe the Perceptron and how it works
- The **Perceptron** is a type of artificial neuron and one of the simplest forms of a neural network. 
- It works by taking multiple binary inputs, applying weights to them, summing them up, and passing the result through an activation function (usually a step function). 
- If the weighted sum exceeds a certain threshold, the Perceptron outputs a 1; otherwise, it outputs a 0.
- It’s used primarily for binary classification problems.

## 2. What is forward and backpropagation for the Perceptron?
- **Forward Propagation**: In this process, the inputs are multiplied by their respective weights, summed up, and passed through the activation function to generate an output. 
- **Backpropagation**: This is a training method used to minimize the error by adjusting the weights. The error between the predicted output and the actual output is calculated, and the weights are updated accordingly in the opposite direction of the error gradient. However, classic Perceptrons don’t use backpropagation, but it's a key feature in multi-layer Perceptrons (MLPs).

## 3. What is the history of the Perceptron?
- The **Perceptron** was developed in 1958 by Frank Rosenblatt, inspired by earlier models like the McCulloch and Pitts neuron.
- It was one of the first algorithms for supervised learning and was implemented in hardware as the Mark I Perceptron.
- Initially, the Perceptron received significant attention, but its limitations (e.g., inability to solve non-linear problems like XOR) were highlighted in the 1960s by Marvin Minsky and Seymour Papert, which led to a decline in neural network research until the resurgence in the 1980s with multi-layer models.

## 4. What is Supervised Training?
- **Supervised Training** is a machine learning approach where the model is trained using labeled data. Each input comes with a corresponding label (desired output).
- The model makes predictions based on the input, and the difference between the predicted and actual labels (error) is used to adjust the model’s weights during training to improve accuracy.

## 5. Why is Perceptron referred to as a binary linear classifier?
- The **Perceptron** is referred to as a **binary linear classifier** because it classifies input data into two categories (binary: 0 or 1) based on a linear decision boundary.
- It uses a linear combination of input features and weights to determine which side of the boundary the input lies, producing either a 0 or 1 as output.

## 6. What are the disadvantages of binary linear classification?
- **Inability to solve non-linear problems**: A Perceptron can only solve problems that are linearly separable. It cannot handle non-linear patterns like the XOR problem.
- **No learning for complex tasks**: A single-layer Perceptron cannot learn complex decision boundaries and is limited to simple tasks. Multi-layer networks are required for more complex functions.
- **Sensitive to noise**: Binary linear classifiers can be sensitive to noisy data, leading to misclassification when input data is not clean or contains outliers.
