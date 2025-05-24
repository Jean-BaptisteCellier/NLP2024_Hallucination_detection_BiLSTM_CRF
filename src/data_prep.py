import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import numpy as np
import unicodedata
from torch.utils.data import DataLoader


def calculate_entropy(logits):
    logits = np.array(logits)
    # Apply softmax to convert logits to probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Stabilize softmax by subtracting max(logits)
    probabilities = exp_logits / exp_logits.sum()
    # Calculate entropy
    probabilities = probabilities[probabilities > 0]  # Exclude zero probabilities
    return -np.sum(probabilities * np.log(probabilities))


def normalize_and_find_with_start(haystack, needle, start=0):
    # Normalize the haystack and needle
    haystack_norm = unicodedata.normalize('NFC', haystack)
    needle_norm = unicodedata.normalize('NFC', needle)
    
    # Perform the find operation starting at the given index
    return haystack_norm.find(needle_norm, start)

def token_overlap(model_input, model_output_ngram):
    input_tokens = set(model_input.split())
    output_tokens = set(model_output_ngram.split())
    if not output_tokens:  # To handle empty output cases
        return 0
    return len(input_tokens & output_tokens) / len(output_tokens)

def map_char_ranges_to_tokens(model_output_text, model_output_tokens):
    """
    Maps character ranges of tokens in `model_output_tokens` to their positions in `model_output_text`.
    """
    token_char_positions = []
    current_char_index = 0

    for token in model_output_tokens:
        # Find the next occurrence of the token in the text
        start_index = normalize_and_find_with_start(model_output_text, token, current_char_index)
        end_index = start_index + len(token)
        token_char_positions.append((start_index, end_index))
        current_char_index = end_index

    return token_char_positions

def get_non_hard_label_texts(row, hard_label_texts):
    model_output_text = row["model_output_text"]
    
    if len(hard_label_texts) == 0:
        return [{
            "text": row["model_output_text"].strip(),
            "start": 0,
            "end": len(row["model_output_text"])-1
        }]

    # Initialize variables
    non_hard_label_texts = []
    # Iterate over hard-labeled ranges to extract non-overlapping parts
    for _index, hard_label in enumerate(hard_label_texts):

        if _index == 0 and hard_label["start"] != 0:
            text_segment = model_output_text[0:hard_label["start"]-1]
            non_hard_label_texts.append({
                "text": text_segment,
                "start": 0,
                "end": hard_label["start"]-1
            })
        if _index + 1 < len(hard_label_texts):
            next_hard_label = hard_label_texts[_index+1]
            text_segment = model_output_text[hard_label["end"]+1:next_hard_label["start"]-1]
            non_hard_label_texts.append({
                "text": text_segment,
                "start": hard_label["end"]+1,
                "end": next_hard_label["start"]-1
            })
        else:
            text_segment = model_output_text[hard_label["end"]+1:]
            non_hard_label_texts.append({
                "text": text_segment,
                "start": hard_label["end"]+1,
                "end": len(row["model_output_text"])-1
            })
       

    non_hard_label_texts = [
        text_info for text_info in non_hard_label_texts 
        if text_info["text"] and text_info["text"] != '.'
    ]
    return non_hard_label_texts

def get_logits_for_text(substring, token_position_dict):
    logits = []
    for token in token_position_dict:
        start_idx = token["startindex"]
        end_idx = token["endindex"]
        if start_idx >= substring["start"] and end_idx <= substring["end"]:
            logits.append(token["logit"])
    substring["logits"] = logits

    return substring

def get_hard_label_texts(row):
    model_output_text = row["model_output_text"]
    hard_labels = row["hard_labels"]
    # Create a list of dictionaries with text and corresponding start and end indices
    hard_label_texts = [
        {
            "text": model_output_text[start:end].strip(),
            "start": start,
            "end": end
        }
        for start, end in hard_labels
        if model_output_text[start:end].strip()  # Ensure non-empty text
    ]

    return hard_label_texts

def build_token_position_dict(text, tokens, logits):
    """
       Build a dictionary mapping tokens to their positions and associated logits.

       This function iterates through the list of tokens and calculates their start and
       end indices in the original text. Each token is paired with its corresponding
       logit value from the model's output.

       Parameters:
       ----------
       text : str
           The original text from which tokens were derived.
       tokens : list of str
           A list of tokens output by the model, potentially normalized or modified.
       logits : list of float
           A list of logit values corresponding to each token.

       Returns:
       -------
       list of dict
           A list of dictionaries, where each dictionary contains:
           - **"token"**: The token text.
           - **"startindex"**: The starting character index of the token in the original text.
           - **"endindex"**: The ending character index of the token in the original text.
           - **"logit"**: The logit value associated with the token.

       Notes:
       -----
       - Relies on a helper function `normalize_and_find_with_start`, which determines the
         start index of a token within the text, accounting for normalization differences.
       - Skips tokens that are empty or whose start index cannot be determined (`-1`).
       - Updates a running index (`current_index`) to ensure correct positioning of tokens in
         the text, especially when tokens are non-contiguous or stripped.

       Examples:
       --------
       >>> text = "The quick brown fox"
       >>> tokens = ["The", "quick", "brown", "fox"]
       >>> logits = [0.8, 0.7, 0.9, 0.6]
       >>> build_token_position_dict(text, tokens, logits)
       [
           {"token": "The", "startindex": 0, "endindex": 3, "logit": 0.8},
           {"token": "quick", "startindex": 4, "endindex": 9, "logit": 0.7},
           {"token": "brown", "startindex": 10, "endindex": 15, "logit": 0.9},
           {"token": "fox", "startindex": 16, "endindex": 19, "logit": 0.6}
       ]
       """
    current_index = 0
    result = []

    # Generate the dictionary
    for idx, token in enumerate(tokens):
        start_index = normalize_and_find_with_start(text, token.strip(), current_index)
        end_index = start_index + len(token.strip())

        if token == '':
            continue
        if start_index == -1:  # Handle cases where token is not in text
            continue

        result.append({
            "token": token,
            "startindex": start_index,
            "endindex": end_index,
            "logit": logits[idx]
        })
        
        current_index = end_index  # Update the current index for the next token

    return result

def generate_ngrams_with_indices(text, start_idx, n):
    """
        Generate n-grams from the input text along with their start and end indices.

        This function splits the input text into words, calculates their positions in the original text,
        and constructs n-grams (sequences of `n` words) with their corresponding start and end indices.
        The indices are adjusted based on the provided `start_idx`.

        Parameters:
        ----------
        text : str
            The input text from which n-grams are to be generated.
        start_idx : int
            The starting index for the text in its larger context.
            This is used to adjust the word positions to match the original text offset.
        n : int
            The size of the n-grams to generate.

        Returns:
        -------
        list of dict
            A list of dictionaries where each dictionary represents an n-gram with:
            - "text": the n-gram text.
            - "start": the start index of the n-gram in the original text.
            - "end": the end index of the n-gram in the original text.

            If the input text has fewer words than `n`, a single dictionary is returned
            with the entire text and its start and end indices.

        Notes:
        -----
        - This function assumes the presence of a helper function `normalize_and_find_with_start`
          which finds the normalized position of a word in the text given a starting index.
        - The function `ngrams` is expected to generate combinations of n-grams from an iterable.

        Examples:
        --------
        >>> text = "The quick brown fox jumps over the lazy dog"
        >>> start_idx = 0
        >>> n = 3
        >>> generate_ngrams_with_indices(text, start_idx, n)
        [
            {'text': 'The quick brown', 'start': 0, 'end': 15},
            {'text': 'quick brown fox', 'start': 4, 'end': 19},
            {'text': 'brown fox jumps', 'start': 10, 'end': 25},
            ...
        ]
        """

    words = text.split()
    word_positions = []
    current_index = 0

    for word in words:
        word_position = normalize_and_find_with_start(text, word, current_index)
        current_index = word_position + len(word)
        word_positions.append((word_position + start_idx, word_position + start_idx + len(word)))


    if len(words) < n:
        return [{"text": text, "start": start_idx, "end": start_idx + len(text)}]
    else:
        ngram_with_indices = []
        for ngram_words in ngrams(enumerate(words), n):
            ngram_start = word_positions[ngram_words[0][0]][0]
            ngram_end = word_positions[ngram_words[-1][0]][1]
            ngram_text = " ".join(word for _, word in ngram_words)
            ngram_with_indices.append({"text": ngram_text, "start": ngram_start, "end": ngram_end})
        return ngram_with_indices
    

def create_bio_tags(data):
    """
    Convert class labels into BIO tags for each token
    """
    bio_tags = []
    previous_tag = "O"
    for _, row in data.iterrows():
        start, label = row["start"], row["class"]
        if label == 1:  # hallucinated token
            if previous_tag == "O":
                bio_tags.append("B") # beginning
            else:
                bio_tags.append("I") # inside
        else:  # non-hallucinated token
            bio_tags.append("O")
        previous_tag = bio_tags[-1]
    return bio_tags


def pad_sequence_features(sequence, max_len, padding_value):
    padded = sequence + [[padding_value] * len(sequence[0])] * (max_len - len(sequence))
    return padded

def pad_sequence_labels(sequence, max_len, padding_value):
    padded = sequence + [padding_value] * (max_len - len(sequence))
    return padded


def prepare_data_for_random_forest_classifier(row, n):
    """
        Prepare data for training or evaluation with a Random Forest classifier.

        This function processes the model's output by generating n-grams, calculating associated
        features (e.g., cosine similarity, token overlap, mean logits, entropy), and combining
        them into a single DataFrame suitable for use with a Random Forest classifier or similar models.

        Parameters:
        ----------
        row : dict
            A dictionary containing various fields related to model output and metadata:
            - "model_output_logits": list of logits corresponding to model's output tokens.
            - "model_output_tokens": list of tokens output by the model.
            - "model_output_text": original text generated by the model.
            - "id": unique identifier for the data point.
            - "model_id": identifier for the model generating the output.
            - "lang": language of the input/output.
            - "model_input": original input to the model.
        n : int
            The size of the n-grams to generate.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the following columns:
            - "id": unique identifier for the data point.
            - "model_id": identifier for the model generating the output.
            - "lang": language of the input/output.
            - "model_input": original input to the model.
            - "model_output_ngram": n-gram text generated from the model output.
            - "ngram_logits": logits associated with the n-gram.
            - "start": start index of the n-gram in the original text.
            - "end": end index of the n-gram in the original text.
            - "class": binary label (0 for non-hard labels, 1 for hard labels).
            - "cosine_similarity": cosine similarity between the input text and the n-gram.
            - "token_overlap": overlap of tokens between the input text and the n-gram.
            - "mean_logits": mean of logits for the n-gram.
            - "entropy_logits": entropy of logits for the n-gram.

        Notes:
        -----
        - Relies on several helper functions:
            - `build_token_position_dict`: Aligns tokens with their positions and logits.
            - `get_hard_label_texts`: Identifies hard-labeled text segments from the model output.
            - `get_non_hard_label_texts`: Identifies non-hard-labeled text segments.
            - `generate_ngrams_with_indices`: Generates n-grams with indices for text.
            - `get_logits_for_text`: Extracts logits for given text from token positions.
            - `cosine_similarity`: Computes cosine similarity between two vectors.
            - `token_overlap`: Computes the token overlap between two strings.
            - `calculate_entropy`: Computes entropy of a probability distribution.
        - Uses `TfidfVectorizer` from scikit-learn to compute cosine similarity.
        ...
    """
    # Tokenize the text and align with logits
    logits = row["model_output_logits"]

    normalized_tokens = [
        token.replace("\u0120", "").replace("\u2581", "").strip() for token in row["model_output_tokens"]
    ]
    token_position_dict = build_token_position_dict(row["model_output_text"], normalized_tokens,logits)
    hard_label_texts = get_hard_label_texts(row)
    non_hard_label_texts = get_non_hard_label_texts(row, hard_label_texts)


    non_hard_label_ngrams = [
        ngram for item in non_hard_label_texts
        for ngram in generate_ngrams_with_indices(item["text"], item["start"], n)
    ]
    hard_label_ngrams = [
        ngram for item in hard_label_texts
        for ngram in generate_ngrams_with_indices(item["text"], item["start"], n)
    ]
    non_hard_label_logits = [get_logits_for_text(ngram, token_position_dict) for ngram in non_hard_label_ngrams]
    hard_label_logits = [get_logits_for_text(ngram, token_position_dict) for ngram in hard_label_ngrams]
    
    ngram_df = pd.DataFrame({
        "id": [row["id"]] * len(non_hard_label_logits),
        "model_id": [row["model_id"]] * len(non_hard_label_logits),
        "lang": [row["lang"]] * len(non_hard_label_logits),
        "model_input": [row["model_input"]] * len(non_hard_label_logits),
        "model_output_ngram": [entry["text"] for entry in non_hard_label_logits],
        "ngram_logits": [entry["logits"] for entry in non_hard_label_logits],
        "start": [entry["start"] for entry in non_hard_label_logits],
        "end": [entry["end"] for entry in non_hard_label_logits],
        "class": [0] * len(non_hard_label_logits)
    })

    hard_label_df = pd.DataFrame({
        "id": [row["id"]] * len(hard_label_logits),
        "model_id": [row["model_id"]] * len(hard_label_logits),
        "lang": [row["lang"]] * len(hard_label_logits),
        "model_input": [row["model_input"]] * len(hard_label_logits),
        "model_output_ngram": [entry["text"] for entry in hard_label_logits],
        "ngram_logits": [entry["logits"] for entry in hard_label_logits],
        "start": [entry["start"] for entry in hard_label_logits],
        "end": [entry["end"] for entry in hard_label_logits],
        "class": [1] * len(hard_label_logits)
    })

    combined_df = pd.concat([ngram_df, hard_label_df], ignore_index=True)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_df["model_input"].tolist() + combined_df["model_output_ngram"].tolist())
    
    input_tfidf = tfidf_matrix[:len(combined_df)]
    ngram_tfidf = tfidf_matrix[len(combined_df):]
    
    cosine_similarities = [
    cosine_similarity(
        input_tfidf[i].reshape(1, -1),
        ngram_tfidf[i].reshape(1, -1)
    )[0, 0]
    for i in range(input_tfidf.shape[0])
    ]
    combined_df["cosine_similarity"] = cosine_similarities
    
    combined_df["token_overlap"] = combined_df.apply(
        lambda x: token_overlap(x["model_input"], x["model_output_ngram"]),
        axis=1
    )

    combined_df['mean_logits'] = combined_df['ngram_logits'].apply(
        lambda x: np.mean(x) if x else None
    )
    combined_df['entropy_logits'] = combined_df['ngram_logits'].apply(
        lambda x: calculate_entropy(x) if x else None
    )
    
    return combined_df


def create_data_loaders(train_data, val_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader