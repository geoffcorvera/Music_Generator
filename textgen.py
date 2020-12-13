import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys
from lstm import LSTM

def make_speech(model, length=1000):
    h = np.zeros((model.n_h, 1))
    c = np.zeros((model.n_h, 1))
    return model.sample(h,c,length)

fp = sys.argv[1]
data = open(fp, 'r', encoding="utf-8").read().lower()
nepochs = int(sys.argv[2])

chars = set(data)
vocab_size = len(chars)
print(f'data has {len(data)} characters, {vocab_size} unique')

char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}

model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs=nepochs, lr=0.001, seq_len=50)
J, params = model.train(data)

# Make speeches
for i in range(5):
    print(f'Generating speech: {i+1}')
    s = make_speech(model)
    fp = path.join('output/speeches-lr001-epoch30', f'speech-{i+1}.txt')
    fout = open(fp, 'w')
    fout.write(s)
    fout.close()

plt.plot([i for i in range(len(J))], J)
plt.xlabel("#training iterations")
plt.ylabel("training loss")
plt.show()