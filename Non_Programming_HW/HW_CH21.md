# Describe Word Embedding
Word embedding is a methodology for mapping discrete words into continuous vector spaces, where each word is represented as a dense vector of fixed dimensions. This representation is learned such that it captures semantic and syntactic relationships between words based on their context in large corpora. Prominent word embedding algorithms include:

1. **Word2Vec**: Implements Skip-Gram and Continuous Bag-of-Words (CBOW) architectures to learn embeddings by predicting surrounding words.
2. **GloVe (Global Vectors for Word Representation)**: Incorporates global statistical information from a corpus to optimize co-occurrence probabilities.
3. **FastText**: Enhances embeddings by considering subword information, effectively capturing morphology.

These embeddings have applications in downstream natural language processing tasks, such as sentiment analysis, named entity recognition, and machine translation.

---

# What is the Measure of Word Similarity?
Word similarity quantifies the semantic closeness between two words based on their embeddings in a vector space. Common similarity measures include:

1. **Cosine Similarity**:
   - Computes the cosine of the angle between two vectors.
   - Formula:  
     \[
     \text{Cosine Similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}
     \]
   - Values range from -1 (opposite) to 1 (identical).

2. **Euclidean Distance**:
   - Measures the straight-line distance between two vectors.
   - Formula:  
     \[
     \text{Euclidean Distance} = \|\vec{A} - \vec{B}\|
     \]

3. **Jaccard Similarity**:
   - Suitable for set-based representations, calculating the ratio of intersecting to union elements.

4. **Dot Product**:
   - Reflects the alignment and magnitude of vectors.

These metrics enable various NLP applications, including synonym detection and clustering.

---

# Describe the Neural Language Model
A Neural Language Model (NLM) predicts the probability of a sequence of words in a given language. Unlike traditional n-gram models, NLMs leverage neural networks to overcome limitations in context length and sparsity. Key components and methodologies include:

1. **Model Architecture**:
   - **Embedding Layer**: Converts input words into dense vectors.
   - **Sequence Modeling**: Uses Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, or Transformers to process sequential dependencies.
   - **Output Layer**: Predicts word probabilities using a softmax function.

2. **Training Objective**:
   - Maximizes the likelihood of the observed sequences or minimizes perplexity, defined as:
     \[
     \text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i | w_1, \ldots, w_{i-1})}
     \]

3. **Applications**:
   - Machine translation, automatic speech recognition, and text generation.

---

# What is a Bias in Word Embedding, and How to Do Debiasing?
**Bias in Word Embedding** refers to the presence of social stereotypes and prejudices encoded in word vector representations, often reflecting underlying biases in the training corpus. For instance, word vectors may associate "doctor" with "male" and "nurse" with "female," perpetuating gender stereotypes.

### Debiasing Techniques
1. **Hard Debiasing**:
   - Identifies the bias direction (e.g., gender axis) through principal component analysis (PCA) and removes this component from word embeddings.
   - Example: Aligns gender-neutral words to be equidistant from gendered terms.

2. **Soft Debiasing**:
   - Reduces the influence of bias while retaining sufficient contextual information.

3. **Counterfactual Data Augmentation**:
   - Balances the training dataset by introducing examples with inverted attributes (e.g., male nurses, female doctors).

4. **Adversarial Training**:
   - Introduces adversarial components to detect and mitigate bias during training.

Debiasing ensures that word embeddings are more equitable and less likely to reinforce harmful stereotypes.

---

# How Does Modern Machine Translation Work Using the Language Model?
Modern machine translation (MT) relies on advanced language models, primarily based on the **sequence-to-sequence (Seq2Seq)** paradigm with attention mechanisms. The current state-of-the-art approach employs Transformer architectures, such as OpenAI's GPT or Google's T5, characterized by the following:

1. **Encoder-Decoder Framework**:
   - **Encoder**: Encodes the source sentence into a latent representation.
   - **Decoder**: Generates the target sentence one token at a time, conditioned on the encoder's output.

2. **Attention Mechanisms**:
   - Allow the model to focus on relevant parts of the input sentence for each word in the output.

3. **Training**:
   - Utilizes bilingual corpora with maximum likelihood estimation or reinforcement learning for fine-tuning.

4. **Pretrained Models**:
   - Models like mBERT and mT5 are pretrained on multilingual data, enabling zero-shot or few-shot translation for low-resource languages.

This methodology significantly improves translation fluency and contextual understanding.

---

# What is Beam Search?
**Beam Search** is a heuristic search algorithm used for decoding in sequence generation tasks, such as machine translation. It explores a limited number of the most probable sequences at each step to balance computational efficiency and quality.

### Algorithm Steps
1. Start with an initial state and generate all possible tokens.
2. Retain the top \( k \) candidates (beam width) based on their cumulative probabilities.
3. For each retained candidate, extend it by generating possible subsequent tokens.
4. Repeat until all sequences reach an end condition (e.g., <eos> token).

### Benefits
- Improves decoding quality by maintaining diversity in hypotheses.
- Avoids exhaustive enumeration of all sequences.

However, beam search may sometimes favor overly generic or repetitive outputs.

---

# What is the BLEU Score?
**BLEU (Bilingual Evaluation Understudy)** is a metric for evaluating the quality of machine-translated text by comparing it with one or more human reference translations. It assesses how well the generated text matches the reference in terms of n-grams.

### Calculation
1. **n-gram Precision**:
   - Measures the proportion of overlapping n-grams between the generated and reference texts.

2. **Brevity Penalty**:
   - Penalizes translations that are significantly shorter than reference translations.

3. **Final Formula**:
   \[
   \text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)
   \]
   Where:
   - \( BP \): Brevity Penalty.
   - \( p_n \): Precision for n-grams.
   - \( w_n \): Weights for each n-gram level.

### Interpretation
- Scores range from 0 to 1 (or 0 to 100%). Higher scores indicate better translation quality but are not absolute measures, as BLEU does not capture semantic equivalence or fluency.