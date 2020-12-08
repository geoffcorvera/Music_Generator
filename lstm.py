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


class LSTM:

    def __init__(self, char_to_idx, idx_to_char, vocab_size, n_h=100, seq_len=25, epochs=10, lr=0.01, beta1=0.9, beta2=0.999):
        self.char_to_idx = char_to_idx      # char/note to indices mapping
        self.idx_to_char = idx_to_char      # indices to char/note mapping
        self.vocab_size = vocab_size        # no. of classes/unique values in training data
        self.n_h = n_h                      # no. units in hidden layer
        self.seq_len = seq_len              # no. of time steps
        self.epochs = epochs                # no. of training iterations
        self.lr = lr                        # learning rate
        self.beta1 = beta1                  # 1st momentum param
        self.beta2 = beta2                  # 2nd momentum param
    
        # Initialise weights & biases
        self.param = {}
        std = (1.0/np.sqrt(self.vocab_size + self.n_h))     # Xavier initialisation

        # Forget gate
        self.params = {}
        self.params["Wf"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bf"] = np.ones((self.n_h, 1))

        # Input gate
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bi"] = np.ones((self.n_h, 1))

        # Cell gate
        self.params["Wc"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bc"] = np.zeros((self.n_h, 1))

        # Output gate
        self.params["Wo"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bo"] = np.zeros((self.n_h, 1))

        # Output
        # XXX: re-examine output activations
        self.params["Wv"] = np.random.randn(self.vocab_size, self.n_h) * (1.0/np.sqrt(self.vocab_size))
        self.params["bv"] = np.zeros((self.vocab_size, 1))

        # Initialise gradients and Adam parameters
        self.grads = {}
        self.adam_params = {}

        for key in self.params:
            self.grads["d"+key] = np.zeros_like(self.params[key])
            self.adam_params["m"+key] = np.zeros_like(self.params[key])
            self.adam_params["v"+key] = np.zeros_like(self.params[key])

        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len

# Configure activation functions to use for LSTM
def sigmoid(self, X):
    return 1 / (1 + np.exp(-X))
    
LSTM.sigmoid = sigmoid


def softmax(self, X):
    e_x = np.exp(X - np.max(X))
    return e_x / np.sum(e_x)

LSTM.softmax = softmax

# TODO: Clip gradients to address exploding gradients?

def reset_gradients(self):
    for key in self.grads:
        self.grads[key].fill(0)

LSTM.reset_grads = reset_gradients

# Update model weights with Adam optimizer
def update_params(self, batch_num):
    for key in self.params:
        self.adam_params["m"+key] = self.adam_params["m"+key] * self.beta1 + (1 - self.beta1) * self.grads["d"+key]
        self.adam_params["v"+key] = self.adam_params["v"+key] * self.beta2 + (1 - self.beta2) * self.grads["d"+key]**2

        m_coorrelated = self.adam_params["m"+key] / (1 - self.beta1**batch_num)
        v_correlated = self.adam_params["v"+key] / (1 - self.beta2**batch_num)
        self.params[key] -= self.lr * m_coorrelated / (np.sqrt(v_correlated) + 1e-8)

LSTM.update_params = update_params

# Create LSTM instance
lstm = LSTM(char_to_idx, idx_to_char, vocab_size)
