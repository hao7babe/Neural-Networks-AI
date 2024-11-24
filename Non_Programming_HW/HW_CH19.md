# RNN Core Concepts and Architecture

## 1. What are recurrent neural networks (RNN) and why are they needed?

RNNs are neural networks designed for processing sequential data with temporal dependencies. Unlike traditional feedforward networks, RNNs have loops that allow information to persist.

They are needed because:
1. Standard neural networks can't handle variable-length sequences
2. They maintain memory of previous inputs
3. They share parameters across time steps, making them efficient

Mathematical formulation:
```
ht = tanh(Whh * ht-1 + Wxh * xt + bh)
output = Why * ht + by
```

## 2. What role do time steps play in recurrent neural networks?

Time steps are crucial elements that:
1. Represent sequential positions in input data
2. Allow information flow through the network
3. Define how far back the network can remember

Example:
```
Sequence: "Hello World"
t=0: "H"
t=1: "e"
t=2: "l"
...and so on
```

## 3. What are the types of recurrent neural networks?

Four main architectures:
1. **One-to-One RNN**
   - Single input, single output
   - Use: Standard classification

2. **One-to-Many RNN**
   - Single input, sequence output
   - Use: Image captioning

3. **Many-to-One RNN**
   - Sequence input, single output
   - Use: Sentiment analysis

4. **Many-to-Many RNN**
   - Sequence input, sequence output
   - Use: Machine translation

## 4. What is the loss function for RNN defined?

RNN loss functions are computed across all time steps:
```
Total Loss = ∑(t=1 to T) Lt(yt_true, yt_pred)
```

For classification tasks:
```
Lt = -∑(c=1 to C) yt_true,c * log(yt_pred,c)
```

## 5. How do forward and backpropagation of RNN work?

### Forward Propagation
```python
for t in range(T):
    # Current hidden state
    ht = tanh(Whh * ht-1 + Wxh * xt + bh)
    # Current output
    yt = Why * ht + by
```

### Backpropagation Through Time (BPTT)
```python
for t in reversed(range(T)):
    # Compute gradients
    dWhy += dyt * ht.T
    dWxh += dht * xt.T
    dWhh += dht * ht-1.T
```

## 6. What are the most common activation functions for RNN?

1. **Tanh Function**
   ```
   tanh(x) = (e^x - e^-x)/(e^x + e^-x)
   Range: [-1, 1]
   ```

2. **Sigmoid Function**
   ```
   σ(x) = 1/(1 + e^-x)
   Range: [0, 1]
   ```

3. **ReLU Function**
   ```
   ReLU(x) = max(0, x)
   Range: [0, ∞)
   ```

## 7. What are bidirectional recurrent neural networks (BRNN) and why are they needed?

BRNNs process sequences in both forward and backward directions simultaneously.

Need for BRNNs:
1. Capture both past and future context
2. Better prediction accuracy for tasks requiring complete sequence context
3. Essential for tasks like speech recognition and translation

Structure:
```
Forward:  →[h1→h2→h3]→
Input:     [x1 x2 x3]
Backward: ←[h1←h2←h3]←
```

## 8. What are Deep recurrent neural networks (DRNN) and why are they needed?

DRNNs stack multiple RNN layers vertically.

Need for DRNNs:
1. Capture more complex patterns in data
2. Learn hierarchical representations
3. Increase model capacity for complex tasks

Architecture:
```
Layer N: [RNN]→[RNN]→[RNN]
Layer 2: [RNN]→[RNN]→[RNN]
Layer 1: [RNN]→[RNN]→[RNN]
Input:    x1    x2    x3
```