# Bidirectional-LSTM-CRF for Hallucination Detection

### Running the Code
To reproduce the results or experiment with the model, you can find the Jupyter Notebook file named **`BiLSTM_CRF.ipynb`** in the repository.

In the first cell, you can define **nb_classes** either as 3 for the "BIO" approach (3-class problem), or as 2 for simple binary classification. The other .py files in the BiLSTM_CRF folder contain the necessary functions to run the notebook, for data processing, scoring, definition of the model, hyperparameter optimization, training and evaluation.


## Overview
- With this approach, our first idea was to **draw inspiration from BIO-tagging in Named Entity Recognition (NER)** and to treat hallucinations like named entities. Thus, each token can be classified in:
  - B: beginning of the hallucination span (encoded as 0)
  - I: inside of a hallucination span (encoded as 1)
  - O: outside of any span, non-hallucinated (encoded as 2)
- Thus, we are considering a 3-class classification problem.
- Data preprocessing and features (token overlap, mean logits...) were derived from the random forest approach.
  - However, we are adding a feature "token_indices" for embedding, giving information about the token's position in the vocabulary.
- Furthermore, we are paying attention to grouping per "sentence" of LLM output, padding all of these sentences to the maximum sequence length and masking of these padding tokens during training and loss computation.
- After experimenting with the "BIO" approach (3 classes), we tried to train the same model with a simple binary classification problem.

### Model Architecture: Bidirectional LSTM-CRF
- The model consists of a Bidirectional LSTM (BiLSTM) layer followed by a Conditional Random Field (CRF) layer. 
  - BiLSTM captures dependencies from both past and future tokens;
  - CRF layer optimizes the sequence labeling by taking into account the most likely sequence of labels. 
This combination allows the model to handle long-term dependencies and make more accurate predictions.

### Training Details
- Hyperparameter tuning with the Optuna library
- Conducted training for **10 epochs** with:
  - Batch size: **8**
  - Embedding dimension: **512**
  - Hidden dimension: **150**
  - Learning rate: **0.005**
  - Optimizer: **Adam**
  However, we often use smaller dimensions to reduce the computational effort.
- Loss function:
  - Used **negative log-likelihood (CRF loss)** to guide the model during training.
- Class Imbalance Handling:
  - Use of **weighted F1-score** (to maximize) to guide the hyperparameter tuning.
- Evaluation Metrics:
  - Accuracy, Precision, Recall, and F1-score.


## Results

### Class Performance

We first notice that the model usually **overfits after a few epochs (3 or 4)**. We do not focus on solving this problem here, and still get acceptable results on the test set:

#### With 3 classes: B, I, O

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0-"B" | 0.52      | 0.14   | 0.22     | 94      |
| 1-"I" | 0.76      | 0.67   | 0.71     | 1780    |
| 2-"O" | 0.79      | 0.88   | 0.83     | 2772    |


#### With 2 classes: 0 (non-hallucinated), 1 (hallucinated)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.88      | 0.84   | 0.84     | 2772    |
| 1     | 0.75      | 0.83   | 0.79     | 1874    |


### Official Scorer Results by Language

#### With 3 classes: B, I, O
| Language | Score  |
|----------|--------|
| fi       | 0.717  |
| hi       | 0.660  |
| de       | 0.621  |
| zh       | 0.614  |
| it       | 0.591  |
| sv       | 0.534  |
| fr       | 0.510  |
| ar       | 0.505  |
| en       | 0.291  |
| es       | 0.197  |


#### With 2 classes
| Language | Score  |
|----------|--------|
| it       | 0.784  |
| hi       | 0.776  |
| fi       | 0.761  |
| zh       | 0.654  |
| de       | 0.606  |
| fr       | 0.573  |
| sv       | 0.566  |
| ar       | 0.499  |
| es       | 0.325  |
| en       | 0.313  |


## Key Insights

- The results show that using three classes in this context might not be very relevant. This indeed introduces a new (and "worst") **class imbalance** between the two types of hallucinated tokens, and class imbalance is difficult to deal with in this context. In particular:
    - The CRF tools in Python don't allow us to specify weights for the different classes.
    - Since we're considering full sequences, data augmentation for class "B" would result in new "I" and "O" instances.
- The official scores obtained with 2 and 3 classes seem very good, and **consistently outperform a random baseline over all languages**. However, with 3 classes, the model is struggling with the minority class "B" (very low recall, f1-score), which explains why this model leads to poorer results than binary classification.