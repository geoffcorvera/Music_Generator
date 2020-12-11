import os, sys
from music21 import converter, instrument, note, chord, stream
import numpy as np
import matplotlib.pyplot as plt
from lstm import LSTM


# Accepts music21 stream object as input.
# Returns dictionary mapping timestep, t, to notes played during t
def get_performance(s):
    allparts = s.flat
    notes = dict()

    # TODO: compare offset, if same, add to list
    for i in allparts:
        t = i.offset
        d = i.duration.quarterLength
        if t not in notes:
            notes[t] = []

        if isinstance(i, note.Note):
            notes[t].append((str(i.pitch), d))
        elif isinstance(i, chord.Chord):
            ps = '.'.join(str(n.pitch) for n in i.notes)
            notes[t].append((ps,d))

    return notes

# Accepts dictionary mapping notes played to their timesteps
# Writes a midi file with provided filename.
def export_midi(performance, filename=None):
    s = stream.Stream()

    for t in performance:
        # Get notes/chords played at timestep t
        notes = performance[t]

        for n in notes:
            if '.' in n[0]:
                ps = [pitch for pitch in n[0].split('.')]
                ch = chord.Chord(ps, quarterLength=n[1])
                s.insert(t, ch)
            else:
                s.insert(t, note.Note(n[0], quarterLength=n[1]))

    if not filename:
        try:
            o_file = sys.argv[1]
            filepath = 'output/' + o_file + '.mid'
            path = s.write('midi', fp=filepath)
        except IndexError:
            path = s.write('midi', fp='output/song.mid')
    else:
        path = s.write('midi', fp='output/'+str(filename))

    print(f"Song written to {path}")

# Returns sequence of notes (Note & Chords)
# XXX: Refactor output to dictionary (time->notes/chords)
def generate_music(model, seed, hidden, state, length=100):
    # Initial hidden activations, cell states, and input vector
    h = np.zeros((model.n_h, 1))
    c = np.zeros((model.n_h, 1))
    x = np.zeros((model.vocab_size, 1))
    id = np.random.choice((range(model.vocab_size)))
    x[id] = 1

    generated_notes = dict()

    # Generate sequence of notes
    for t in range(length):
        y_hat, _, h, _, c, _, _, _, _ = model.forward_step(x, h, c)

        # Select note to play
        idx = np.random.choice(range(model.vocab_size), p=y_hat.ravel())
        # Create input for next iteration
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1

        generated_notes[t](model.idx_to_char[idx])
    
    return generated_notes

def note_sequence(d):
    n = []
    for t in d:
        notes = d[t]
        n.append(notes)
    return n

flatten = lambda t: [item for sublist in t for item in sublist]

# Extract notes and durations from midi files
folder = 'data/test'
notes = []
for f in os.listdir(folder):
    fp = os.path.join(folder, f)
    midi_stream = converter.parse(fp)
    perf = get_performance(midi_stream)
    ns = note_sequence(perf)
    notes += ns

# Get set of unique notes and durations
unique_notes = set(n for n in flatten(notes))
n_vocab = len(unique_notes)

# Create mappings between <note,int> and <duration,int>
note_to_int = dict((nt, num) for num, nt in enumerate(unique_notes))
int_to_note = dict((num, nt) for num, nt in enumerate(unique_notes))

# XXX: Construct input vectors


model = LSTM(note_to_int, int_to_note, n_vocab, epochs=10, lr=0.01)
error, params = model.train(notes, verbose=False)

# Output trained model parameters
for key in params:
    filename = f"output/{str(key)}.csv"
    print(f'output {str(key)} to "output/{filename}"')
    np.savetxt(filename, params[key], delimiter=",")

plt.plot([i for i in range(len(error))], error)
plt.xlabel('#training iterations')
plt.ylabel('training loss')
plt.show()

# Create initial states for model
hidden = np.zeros((model.n_h, 1))
state = np.zeros((model.n_h, 1))
# Select one note to be "On" for initial input
seed = np.zeros((model.vocab_size, 1))
id = np.random.choice((range(model.vocab_size)))
seed[id] = 1

song = generate_music(model, seed, hidden, state, length=512)
export_midi(song)

        
