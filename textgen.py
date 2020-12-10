from lstm import LSTM
import matplotlib.pyplot as plt

data = open('data/hitchiker/guide.txt', 'r', encoding='utf-8').read().lower()

chars = set(data)
vocab_size = len(chars)
print(f'data has {len(data)} characters, {vocab_size} unique')

char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}

model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs=5, lr=0.01)
J, params = model.train(data)

plt.plot([i for i in range(len(J))], J)
plt.xlabel("#training iterations")
plt.ylabel("training loss")