# Pooling Layer and How It Works

A **pooling layer** in a convolutional neural network (CNN) is used to reduce the spatial dimensions (width and height) of the input feature map. This helps to:
- Reduce computational complexity.
- Extract dominant features and make the network invariant to small translations.

### How It Works
Pooling works by applying a filter over the input feature map and performing an aggregation operation (e.g., max or average) over the region covered by the filter.

#### Types of Pooling:
1. **Max Pooling**: Selects the maximum value within the filter region.
2. **Average Pooling**: Computes the average of the values in the filter region.
3. **Global Pooling**: Reduces the entire feature map to a single value by applying pooling over all elements.

---

# Three Major Types of Layers in a Convolutional Neural Network

1. **Convolutional Layer**:
   - Applies filters to extract spatial features such as edges, textures, and shapes.
   - Includes learnable parameters (filters) and usually uses an activation function like ReLU.

2. **Pooling Layer**:
   - Downsamples feature maps to reduce their spatial dimensions.
   - Reduces computational load and provides translation invariance.

3. **Fully Connected (Dense) Layer**:
   - Flattens the feature maps into a single vector and connects every neuron in one layer to every neuron in the next.
   - Used for high-level decision-making and classification.

---

# Architecture of a Convolutional Neural Network

A typical CNN architecture consists of the following layers in sequence:

1. **Input Layer**:
   - Receives raw input data such as images (e.g., \(32 \times 32 \times 3\) for RGB images).

2. **Convolutional Layer(s)**:
   - Applies filters to extract features.
   - Uses activation functions (e.g., ReLU) to introduce non-linearity.

3. **Pooling Layer(s)**:
   - Reduces spatial dimensions and retains the most important features.

4. **Normalization Layer (Optional)**:
   - Stabilizes learning by normalizing activations (e.g., Batch Normalization).

5. **Fully Connected Layer(s)**:
   - Flattens the feature maps and connects neurons for classification or regression tasks.

6. **Output Layer**:
   - Produces the final predictions, typically with a softmax or sigmoid function.

---

### Example CNN Architecture
- **Input Layer**: \(32 \times 32 \times 3\) RGB image.
- **Conv Layer 1**: \(32 \times 32 \times 16\) (16 filters, \(3 \times 3\) kernel).
- **Pooling Layer 1**: \(16 \times 16 \times 16\) (Max Pooling, \(2 \times 2\)).
- **Conv Layer 2**: \(16 \times 16 \times 32\) (32 filters).
- **Pooling Layer 2**: \(8 \times 8 \times 32\).
- **Fully Connected Layer**: \(1 \times 128\) neurons.
- **Output Layer**: \(1 \times 10\) neurons (for 10-class classification).