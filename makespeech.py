from lstm import LSTM
from os import path, listdir
import sys
import numpy as np

def load_params(folder):
    params = {}
    prefix = folder
    for f in listdir(prefix):
        key = f.split('.')[0]
        params[key] = np.genfromtxt(path.join(prefix, f"{key}.csv"), delimiter=',')
    return params

def make_speech(model, length=1000):
    h = np.zeros((model.n_h, 1))
    c = np.zeros((model.n_h, 1))
    return model.sample(h,c,length)

def create_mappings(X):
    uniques = set(X)
    vocab_size = len(uniques)
    char_to_idx = {w: i for i,w in enumerate(uniques)}
    idx_to_char = {i: w for i,w in enumerate(uniques)}
    return vocab_size, char_to_idx, idx_to_char

fp = sys.argv[1]
data = open(fp, 'r', encoding="utf-8").read().lower()
# Create mappings between ints and chars
vocab_size, char_to_idx, idx_to_char = create_mappings(data)
print(f'data has {len(data)} characters, {vocab_size} unique')
# Load learned parameters from previous training rounds
P = load_params('training')
# Initialize model with learned parameters
model = LSTM(char_to_idx, idx_to_char, vocab_size, seq_len=50, params=P)

new_speech = make_speech(model)

# Get output filename
fname = sys.argv[2]
fout = open(fname, 'w', encoding='utf-8')
fout.write(new_speech)
fout.close()