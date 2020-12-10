import os, sys
from music21 import converter, instrument, note, chord, stream
import numpy as np
import matplotlib.pyplot as plt
from lstm import LSTM


# Use music21 to parse midi files for notes and chords
def getAllSongs():
    notes = []
    for curr, _, files in os.walk('data'):
        for f in files:
            file = os.path.join(curr, f)
            midi = converter.parse(file)

            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            
            for item in notes_to_parse:
                if isinstance(item, note.Note):
                    notes.append(str(item.pitch))
                elif isinstance(item, chord.Chord):
                    notes.append('.'.join(str(n) for n in item.normalOrder))
    
    return notes


def parseMidi(dir, filename):
    f = os.path.join(dir, filename)
    return converter.parse(f)

# Returns list of notes and list of durations
def extract_notes(s):

    as_chords = s.chordify()
    notes = []

    for i in as_chords.recurse():
        if isinstance(i, note.Note):
            p = str(i.pitch)
            notes.append((p, i.duration.quarterLength))
        elif isinstance(i, chord.Chord):
            ps = '.'.join(str(n.pitch) for n in i.notes)
            notes.append((ps, i.duration.quarterLength))

    return notes


# Returns sequence of notes (Note & Chords)
def generate_music(model, seed, hidden, state, length=100):
    # Initial hidden activations, cell states, and input vector
    h = np.zeros((model.n_h, 1))
    c = np.zeros((model.n_h, 1))
    x = np.zeros((model.vocab_size, 1))
    id = np.random.choice((range(model.vocab_size)))
    x[id] = 1

    generated_notes = []

    # Generate sequence of notes
    for _ in range(length):
        y_hat, _, h, _, c, _, _, _, _ = model.forward_step(x, h, c)

        # Select note to play
        idx = np.random.choice(range(model.vocab_size), p=y_hat.ravel())
        # Create input for next iteration
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1

        generated_notes.append(model.idx_to_char[idx])
    
    return generated_notes


# Write midi file from list notes & chords
def export_midi(notes):
    s = stream.Stream()
    for nt in notes:
        if '.' in nt[0]:
            ps = [pitch for pitch in nt[0].split('.')]
            s.append(chord.Chord(ps, quarterLength=nt[1]))
        else:
            s.append(note.Note(nt[0], quarterLength=nt[1]))

    try:
        o_file = sys.argv[1]
        filepath = 'output/' + o_file + '.mid'
        path = s.write('midi', fp=filepath)
    except IndexError:
        path = s.write('midi', fp='output/song.mid')

    print(f"Song written to {path}")

    
# Extract notes and durations from midi files
folder = 'data/test'
notes = []
for f in os.listdir(folder):
    fp = os.path.join(folder, f)
    midi_stream = converter.parse(fp)
    ns = extract_notes(midi_stream)
    notes += ns

# Get set of unique notes and durations
unique_notes = set(n for n in notes)
n_vocab = len(unique_notes)

# notes = processTestMidi()
# pitchnames = sorted(set(item for item in notes))
# n_pitches = len(pitchnames)

# Create mappings between <note,int> and <duration,int>
note_to_int = dict((nt, num) for num, nt in enumerate(unique_notes))
int_to_note = dict((num, nt) for num, nt in enumerate(unique_notes))

model = LSTM(note_to_int, int_to_note, n_vocab, epochs=75, lr=0.01)
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

song = generate_music(model, seed, hidden, state, length=256)
export_midi(song)

        
