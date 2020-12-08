import numpy as np
import matplotlib.pyplot as plt

data = open('data/austen-pride-and-prejudice.txt', 'r', encoding='utf-8').read().lower()

chars = set(data)
vocab_size = len(chars)
print(f'data has {len(data)} characters, {vocab_size} unique')

char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}

# "Adam" optimiser
# Xavier initialisation (involves random sampling)
# Create LSTM input vectors (dimensions [vocab_size + n_h, 1])
# LSTM layer wants to ouptu n_h neurons, each weight should be size [n_h, vocab_size + n_h]
    # and each bias of size [n_h, 1]


# Activation Layer
    # each category is a binary switch (thresholding at 0.5) to allow for multiple notes to be on at a time
# 