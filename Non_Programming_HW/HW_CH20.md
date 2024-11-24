# Understanding Language Models and Advanced RNN Architectures

## 1. How does language model (LM) work?

A language model predicts the probability of a sequence of words or the next word given previous words.

Mathematical formulation:
```
P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * P(w3|w1,w2) * ... * P(wn|w1,...,wn-1)
```

Key components:
1. Vocabulary creation
2. Token embedding
3. Context processing
4. Probability distribution generation

## 2. How does word prediction work?

Word prediction uses conditional probability to predict the next word:

```
P(next_word|previous_words) = softmax(Wx + b)
```

Process:
1. Input processing: Convert words to tokens
2. Context encoding: Process previous words
3. Probability calculation: Generate probabilities for each possible next word
4. Selection: Choose highest probability word

Example:
```
Input: "The cat sits on the"
Predictions: 
- "mat" (0.3)
- "floor" (0.25)
- "chair" (0.2)
```

## 3. How to train an LM?

Language Model training process:

1. **Data Preparation**
```python
# Convert text to sequences
input_sequence = words[:-1]
target_sequence = words[1:]
```

2. **Training Steps**
```python
# For each sequence:
1. Forward pass: compute predictions
2. Calculate loss: cross-entropy
3. Backpropagate: update weights
4. Optimize: using methods like Adam
```

3. **Loss Function**
```
Loss = -∑ y_true * log(y_pred)
```

## 4. Describe the problem and the nature of vanishing and exploding gradients

### Vanishing Gradients
- Problem: Gradients become extremely small during backpropagation
```
Gradient ≈ 0 as it propagates back
∏(∂h/∂t) → 0
```

### Exploding Gradients
- Problem: Gradients become extremely large
```
Gradient → ∞ as it propagates back
∏(∂h/∂t) → ∞
```

Causes:
1. Deep network depth
2. Activation function derivatives
3. Weight matrix properties

## 5. What is LSTM and the main idea behind it?

LSTM (Long Short-Term Memory) is designed to address the vanishing gradient problem through gating mechanisms.

Core equations:
```
ft = σ(Wf[ht-1, xt] + bf)  # Forget gate
it = σ(Wi[ht-1, xt] + bi)  # Input gate
ot = σ(Wo[ht-1, xt] + bo)  # Output gate
ct = ft * ct-1 + it * tanh(Wc[ht-1, xt] + bc)  # Cell state
ht = ot * tanh(ct)  # Hidden state
```

Main components:
1. Forget gate: Controls information to discard
2. Input gate: Controls new information to store
3. Output gate: Controls information output
4. Cell state: Long-term memory storage

## 6. What is GRU?

GRU (Gated Recurrent Unit) is a simplified version of LSTM with fewer parameters.

Mathematical formulation:
```
zt = σ(Wz[ht-1, xt] + bz)  # Update gate
rt = σ(Wr[ht-1, xt] + br)  # Reset gate
h̃t = tanh(W[rt * ht-1, xt] + b)  # Candidate hidden state
ht = (1 - zt) * ht-1 + zt * h̃t  # Final hidden state
```

Key differences from LSTM:
1. Combines forget and input gates into update gate
2. Merges cell state and hidden state
3. Introduces reset gate for short-term memory control

Advantages:
1. Fewer parameters to train
2. Similar performance to LSTM
3. More efficient computation