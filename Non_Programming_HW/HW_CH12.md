### 1. What is the reason for softmax?

The **reason for using softmax** is to convert raw model outputs (called logits) into probabilities that sum to 1, making it easier to interpret the model’s predictions in multi-class classification tasks. Each class is assigned a probability that reflects how likely it is the correct class, allowing the model to make clear predictions.

### 2. What is softmax and how does it work?

- **What is Softmax**: Softmax is a mathematical function that transforms a vector of raw scores (logits) into a probability distribution. It is commonly used as the final activation function in neural networks for multi-class classification.

- **How Softmax Works**:
  1. For each class \(i\), the softmax function computes the exponential of the input logits \(z_i\) (to make them positive).
  2. It then normalizes these exponentials by dividing by the sum of all exponentials for all classes. This ensures that the output probabilities sum to 1.
  
The formula for softmax is:

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

Where:
- \(z_i\) is the raw score (logit) for class \(i\),
- \(e^{z_i}\) is the exponentiation of the logit for class \(i\),
- The denominator sums the exponentials for all classes \(j\).

- **Interpretation**: After applying softmax, the output values represent the probability that a given input belongs to each class, with the highest probability indicating the predicted class.

In summary, softmax turns raw scores into a meaningful probability distribution, making it ideal for tasks where we need to classify inputs into one of multiple categories.