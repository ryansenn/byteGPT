import numpy as np
import random


with open("data.txt", "r") as file:
    data = file.read().replace("\n", "")

tokens = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!'’-+():\"\\“\\”\n"

# Create character-to-index and index-to-character mappings
char_to_idx = {ch: i for i, ch in enumerate(tokens)}
idx_to_char = {i: ch for i, ch in enumerate(tokens)}


# Build bigram probability matrix
def build_bigram_probabilities(text):
    n_tokens = len(tokens)
    bigram_count = np.ones((n_tokens, n_tokens))

    for i in range(len(text) - 1):
        bigram_count[char_to_idx[text[i]]][char_to_idx[text[i + 1]]] += 1

    # Convert counts to probabilities
    row_sums = bigram_count.sum(axis=1, keepdims=True)
    return bigram_count / row_sums


bigram_prob = build_bigram_probabilities(data)


# Sample the next token based on probabilities
def next_token(prev):
    prev_index = char_to_idx[prev]
    return idx_to_char[np.random.choice(len(tokens), p=bigram_prob[prev_index])]


# Generate text
def generate(start='h', length=100):
    result = [start]
    curr = start

    for _ in range(length - 1):
        curr = next_token(curr)
        result.append(curr)

    return ''.join(result)


# Generate and print text
print(generate())