import os
from music21 import converter, instrument, note, chord, midi


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

def open_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)


def getDebussy():
    notes = []
    prefix = 'data/debussy'
    for f in os.listdir(prefix):
        file = os.path.join(prefix, f)
        midi_stream = open_midi(file)
        notes = [nt.pitch for nt in midi_stream.flat.notes if isinstance(nt, note.Note)]

    return notes

    
# Extract notes from Debussy folder
notes = getDebussy()
pitchnames = sorted(set(item for item in notes))
n_pitches = len(pitchnames)

# Make dictionary to map note pitches to integers
note_to_int = dict((nt, num) for num, nt in enumerate(pitchnames))

network_input = []
network_output = []
