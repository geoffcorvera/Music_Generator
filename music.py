import os, sys
from music21 import converter, note, chord, stream
from music21 import instrument

# Write midi file from list notes & chords
def export_midi(notes):
    s = stream.Stream()
    for nt in notes:
        if '.' in nt:
            s.append(chord.Chord([pitch for pitch in nt.split('.')]))
        else:
            s.append(note.Note(nt, type='quarter'))

    s.show()
    try:
        o_file = sys.argv[1]
        filepath = 'output/' + o_file + '.mid'
        path = s.write('midi', fp=filepath)
    except IndexError:
        path = s.write('midi', fp='output/song.mid')

    print(f"Song written to {path}")


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


export_midi(notes)