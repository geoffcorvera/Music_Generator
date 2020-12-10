import os
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
    """
        parse file > .chordify
        extract notes from it
        If note.Note
        elif note.Chord
    """

    as_chords = s.chordify()
    notes = []
    durations = []

    for i in as_chords.recurse():
        if isinstance(i, note.Note):
            notes.append(str(i.pitch))
            durations.append(i.duration.quarterLength)
        elif isinstance(i, chord.Chord):
            notes.append('.'.join(str(n.pitch) for n in i.notes)) 
            durations.append(i.duration.quarterLength)

    return notes, durations


def processTestMidi():
    notes = []
    prefix = 'data/test'
    for f in os.listdir(prefix):
        file = os.path.join(prefix, f)
        midi_stream = converter.parse(file)

        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi_stream)
        
        if parts:
            notes_to_parse = parts.recurse()
        else:
            notes_to_parse = midi_stream.flat.notes
            # notes_to_parse = [nt.pitch for nt in midi_stream.flat.notes if isinstance(nt, note.Note)]

        for item in notes_to_parse:
            if isinstance(item, note.Note):
                notes.append(str(item.pitch))
            elif isinstance(item, chord.Chord):
                notes.append('.'.join(str(n.pitch) for n in item.notes))

    return notes

    
# Extract notes and durations from midi files
folder = 'data/test'
notes = []
durations = []
for f in os.listdir(folder):
    fp = os.path.join(folder, f)
    midi_stream = converter.parse(fp)
    ns, ds = extract_notes(midi_stream)
    notes += ns
    durations += ds

# Get set of unique notes and durations
unique_notes = set(n for n in notes)
n_vocab = len(unique_notes)
unique_durations = set(d for d in durations)
n_durations = len(unique_durations)

# notes = processTestMidi()
# pitchnames = sorted(set(item for item in notes))
# n_pitches = len(pitchnames)

# Create mappings between <note,int> and <duration,int>
note_to_int = dict((nt, num) for num, nt in enumerate(unique_notes))
int_to_note = dict((num, nt) for num, nt in enumerate(unique_notes))
dur_to_int = dict((d,i) for i,d in enumerate(unique_durations))
int_to_dur = dict((i,d) for i,d in enumerate(unique_durations))

model = LSTM(note_to_int, int_to_note, n_vocab, epochs=100, lr=0.01)
error, params = model.train(notes)

# Output trained model parameters
# TODO: test
for key in params:
    filename = f"output/{str(key)}.csv"
    print(f'output {str(key)} to "output/{filename}"')
    np.savetxt(filename, params[key], delimiter=",")

plt.plot([i for i in range(len(error))], error)
plt.xlabel('#training iterations')
plt.ylabel('training loss')
plt.show()

# Returns sequence of notes (Note & Chords)
def generate_music(model, seed, hidden, state):
    # Initial hidden activations, cell states, and input vector
    h = np.zeros((model.n_h, 1))
    c = np.zeros((model.n_h, 1))
    x = np.zeros((model.vocab_size, 1))
    id = np.random.choice((range(model.vocab_size)))
    x[id] = 1

    sample_size=100
    generated_notes = []

    # Generate sequence of notes
    for _ in range(sample_size):
        y_hat, _, h, _, c, _, _, _, _ = model.forward_step(x, h, c)

        # Select note to play
        idx = np.random.choice(range(model.vocab_size), p=y_hat.ravel())
        # Create input for next iteration
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1

        generated_notes.append(model.idx_to_char[idx])
    
    return generated_notes


# Create initial states for model
hidden = np.zeros((model.n_h, 1))
state = np.zeros((model.n_h, 1))
# Select one note to be "On" for initial input
seed = np.zeros((model.vocab_size, 1))
id = np.random.choice((range(model.vocab_size)))
seed[id] = 1

generated_notes = generate_music(model, seed, hidden, state)

# Write midi file from notes
s = stream.Stream()
for nt in generated_notes:
    if '.' in nt:
        s.append(chord.Chord([pitch for pitch in nt.split('.')]))
    else:
        s.append(note.Note(nt, type='quarter'))

s.show()
path = s.write('midi', fp='output/song.mid')
print(f"Song written to {path}")

