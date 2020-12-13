"""
Adapted from Christina Kouridi's blog post "Implementing a LSTM from scratch with Numpy":
https://christinakouridi.blog/2019/06/20/vanilla-lstm-numpy/

"""
import numpy as np
import matplotlib.pyplot as plt



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
    
    # Forward propagation for time-step
    def forward_step(self, x, h_prev, c_prev):
        z = np.row_stack((h_prev, x))

        f = self.sigmoid(np.dot(self.params["Wf"], z) + self.params["bf"])
        i = self.sigmoid(np.dot(self.params["Wi"], z) + self.params["bi"])
        c_bar = np.tanh(np.dot(self.params["Wc"], z) + self.params["bc"])

        c = f * c_prev + i * c_bar
        o = self.sigmoid(np.dot(self.params["Wo"], z) + self.params["bo"])
        h = o * np.tanh(c)

        v = np.dot(self.params["Wv"], h) + self.params["bv"]
        y_hat = self.softmax(v)
        return y_hat, v, h, o, c, c_bar, i, f, z
    

    # Back propagation for time-step
    def back_step(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
        dv = np.copy(y_hat)
        dv[y] -= 1 # yhat - y

        self.grads["dWv"] += np.dot(dv, h.T)
        self.grads["dbv"] += dv

        dh = np.dot(self.params["Wv"].T, dv)
        dh += dh_next

        do = dh * np.tanh(c)
        da_o = do * o*(1-o)
        self.grads["dWo"] += np.dot(da_o, z.T)
        self.grads["dbo"] += da_o

        dc = dh * o * (1-np.tanh(c)**2)
        dc += dc_next

        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar**2)
        self.grads["dWc"] += np.dot(da_c, z.T)
        self.grads["dbc"] += da_c

        di = dc * c_bar
        da_i = di * i*(1-i)
        self.grads["dWi"] += np.dot(da_i, z.T)
        self.grads["dbi"] += da_i

        df = dc * c_prev
        da_f = df * f*(1-f)
        self.grads["dWf"] += np.dot(da_f, z.T)
        self.grads["dbf"] += da_f

        dz = (np.dot(self.params["Wf"].T, da_f)
            + np.dot(self.params["Wi"].T, da_i)
            + np.dot(self.params["Wc"].T, da_c)
            + np.dot(self.params["Wo"].T, da_o))
        
        dh_prev = dz[:self.n_h, :]
        dc_prev = f * dc
        return dh_prev, dc_prev

    # Executes forward and backward steps, iterating over all time-steps
    # Cross entropy loss is calculated in forward phase.
    # forward_backward exports cross entropy loss of the training batch,
    # and hidden & cell states of the last layer (to feed into next LSTM
    # input layer for next batch)
    def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
        x, z = {}, {}
        f, i, c_bar, c, o = {},{},{},{},{}
        y_hat, v, h = {},{},{}

        # Values at t=-1
        h[-1] = h_prev
        c[-1] = c_prev

        loss = 0
        for t in range(self.seq_len):
            x[t] = np.zeros((self.vocab_size, 1))
            x[t][x_batch[t]] = 1

            y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = self.forward_step(x[t], h[t-1], c[t-1])

            loss += -np.log(y_hat[t][y_batch[t],0])
        
        self.reset_grads()

        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(self.seq_len)):
            dh_next, dc_next = self.back_step(y_batch[t], y_hat[t], dh_next,
                                dc_next, c[t-1], z[t], f[t], i[t],
                                c_bar[t], c[t], o[t], h[t])
        
        return loss, h[self.seq_len-1], c[self.seq_len-1]
    

    # Takes as input sequence of notes (X) and outputs a list of
    # losses for each training batch (J) as well as the trained parameters
    def train(self, X, verbose=True):
        J = []  # to store losses
        # use floor division to round to nearest whole number
        num_batches = len(X) // self.seq_len
        X_trimmed = X[:num_batches * self.seq_len]

        for epoch in range(self.epochs):
            h_prev = np.zeros((self.n_h, 1))
            c_prev = np.zeros((self.n_h, 1))

            for j in range(0, len(X_trimmed) - self.seq_len, self.seq_len):
                # prepare batches
                x_batch = [self.char_to_idx[ch] for ch in X_trimmed[j: j+self.seq_len]]
                y_batch = [self.char_to_idx[ch] for ch in X_trimmed[j + 1: j + self.seq_len + 1]]

                loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)

                # smooth out loss and store in list
                self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                J.append(self.smooth_loss)
                
                self.clip_grads()

                batch_num = epoch * self.epochs + j / self.seq_len + 1
                self.update_params(batch_num)

                # Print loss and sample string
                if verbose:
                    if j % 400000 == 0:
                        print(f"Epoch: {epoch}\tBatch: {j}-{j+self.seq_len}\tLoss {round(self.smooth_loss, 2)}")
                        s = self.sample(h_prev, c_prev, sample_size=250)
                        print(s)

        return J, self.params

    


# Configure activation functions to use for LSTM
def sigmoid(self, X):
    return 1 / (1 + np.exp(-X))
    
LSTM.sigmoid = sigmoid

# XXX: Does this enable having multiple notes on at a time?
def thresholding(self, X):
    activation = self.sigmoid(X)
    return np.where(activation > 0.5, 1, 0)

LSTM.thresholding = thresholding

def softmax(self, X):
    e_x = np.exp(X - np.max(X))
    return e_x / np.sum(e_x)

LSTM.softmax = softmax

# TODO: Clip gradients to address exploding gradients?
def clip_grads(self):
    for key in self.grads:
        np.clip(self.grads[key], -5, 5, out=self.grads[key])

LSTM.clip_grads = clip_grads

def reset_grads(self):
    for key in self.grads:
        self.grads[key].fill(0)

LSTM.reset_grads = reset_grads

# Update model weights with Adam optimizer
def update_params(self, batch_num):
    for key in self.params:
        self.adam_params["m"+key] = self.adam_params["m"+key] * self.beta1 + (1 - self.beta1) * self.grads["d"+key]
        self.adam_params["v"+key] = self.adam_params["v"+key] * self.beta2 + (1 - self.beta2) * self.grads["d"+key]**2

        m_coorrelated = self.adam_params["m"+key] / (1 - self.beta1**batch_num)
        v_correlated = self.adam_params["v"+key] / (1 - self.beta2**batch_num)
        self.params[key] -= self.lr * m_coorrelated / (np.sqrt(v_correlated) + 1e-8)

LSTM.update_params = update_params

def sample(self, h_prev, c_prev, sample_size):
    x = np.zeros((self.vocab_size, 1))
    h = h_prev
    c = c_prev
    sample_string = ""

    for _ in range(sample_size):
        y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)

        idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
        x = np.zeros((self.vocab_size, 1))
        x[idx] = 1

        charNote = self.idx_to_char[idx]
        sample_string += charNote
    
    return sample_string

LSTM.sample = sample