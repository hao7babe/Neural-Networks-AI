### 1. **Hadamard Matrix Product**
The Hadamard product, or element-wise multiplication, is when you multiply two matrices of the same size by multiplying their corresponding elements. Imagine you have two grids of numbers with the same dimensions. For each position in the grid, you just multiply the numbers at that spot together to get a new grid. It’s a simple, straightforward multiplication that happens element by element.

### 2. **Matrix Multiplication**
Matrix multiplication is a more complex operation where you combine two matrices in a specific way. Instead of multiplying elements directly, you take the rows of the first matrix and the columns of the second matrix, multiply them pairwise, and sum them up to get each element of the resulting matrix. It’s like combining information from both matrices into a new one, but only works when the number of columns in the first matrix matches the number of rows in the second.

### 3. **Transpose Matrix and Vector**
- **Matrix Transpose**: A transpose of a matrix is like flipping it over its diagonal. This means the rows become columns and the columns become rows. If you imagine the matrix as a table, the first row becomes the first column, the second row becomes the second column, and so on.
- **Vector Transpose**: If you have a column vector (a list going down), transposing it turns it into a row vector (a list going across), and vice versa.

### 4. **Training Set Batch**
A batch is a small subset of the training data used in one iteration of training a neural network. Instead of feeding the entire dataset into the model at once, which can be slow and resource-intensive, we split the data into smaller batches. This way, the model learns from small pieces of data at a time, which helps it train faster and use less memory.

### 5. **Entropy-based Loss Function**
An entropy-based loss function, like cross-entropy, measures how well a model’s predictions match the actual outcomes. It’s especially useful for classification tasks where you’re predicting probabilities. The idea is to calculate how far off the model’s predicted probability distribution is from the actual distribution. The smaller the loss, the better the model’s predictions are. It helps the model learn to make predictions that are more and more like the actual answers.

### 6. **Neural Network Supervised Training Process**
Training a neural network involves teaching it to make predictions based on examples where the answers are already known. Here’s a simplified version of the process:
1. **Initialization**: Start with random weights, like guessing.
2. **Forward Pass**: Feed the input data through the network to get predictions.
3. **Calculate Loss**: Compare the predictions to the actual answers using a loss function to see how far off they are.
4. **Backpropagation**: Send the error information backward through the network to figure out how to adjust each weight to reduce the error.
5. **Weight Update**: Change the weights slightly based on the error information, using an optimization algorithm like gradient descent.
6. **Repeat**: Keep doing this for many iterations until the model gets good at making predictions.

### 7. **Forward Propagation and Backpropagation**
- **Forward Propagation**: This is the process where you pass your input data through the network from the input layer to the output layer. Each layer processes the data, making it a little closer to the final prediction. It’s like a series of filters or transformations.
  
- **Backpropagation**: After you’ve seen how far off your predictions are (using the loss function), you work backward through the network to see how much each weight contributed to the error. Then, you adjust those weights to make the predictions better next time. It’s like figuring out where you went wrong in solving a math problem and fixing each step to get the correct answer.