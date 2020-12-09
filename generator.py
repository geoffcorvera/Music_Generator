import os
from music21 import converter, instrument, note, chord, midi
import numpy as np
import matplotlib.pyplot as plt
from lstm import LSTM


# Use music21 to parse midi files for notes and chords
# XXX: Run on Google Cloud (free trial)
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

# use 1-hot-encoding for categorical data (notes)
def oneHotEncoding(values, nclasses):
    return np.eye(nclasses)[values]


def open_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)


def processTestMidi():
    notes = []
    prefix = 'data/test'
    for f in os.listdir(prefix):
        file = os.path.join(prefix, f)
        midi_stream = open_midi(file)
        notes = [nt.pitch for nt in midi_stream.flat.notes if isinstance(nt, note.Note)]

    return notes

    
# Extract notes from Debussy folder
notes = processTestMidi()
pitchnames = sorted(set(item for item in notes))
n_pitches = len(pitchnames)

# Make dictionary to map note pitches to integers, and back
note_to_int = dict((nt, num) for num, nt in enumerate(pitchnames))
int_to_note = dict((num, nt) for num, nt in enumerate(pitchnames))

model = LSTM(note_to_int, int_to_note, n_pitches, epochs=100, lr=0.01)
error, params = model.train(notes)

# Output trained model parameters
# TODO: test
for key in params:
    filename = f"{str(key)}.csv"
    print(f'output {str(key)} to "output/{filename}"')
    np.savetxt(filename, params[key], delimiter=",")

plt.plot([i for i in range(len(error))], error)
plt.xlabel('#training iterations')
plt.ylabel('training loss')
plt.show()


def test():
    # TODO: experiment with different sequence lengths
    seq_len = 100

    network_input = []
    network_output = []

    # Create input sequences
    for i in range(len(notes) - seq_len):
        seq_in = notes[i:i+seq_len]
        seq_out = notes[i+seq_len]
        network_input.append([note_to_int[nt] for nt in seq_in])
        network_output.append(note_to_int[seq_out])

    n_sequences = len(network_input)
    # Reshape input to LSTM compatible format
    network_input = np.reshape(network_input, (n_sequences, seq_len, 1))
    # Normalize input
    network_input = network_input /float(n_pitches)
    network_output = oneHotEncoding(network_output, n_pitches)


"""
data = open('data/austen-pride-and-prejudice.txt', 'r', encoding='utf-8').read().lower()

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
"""