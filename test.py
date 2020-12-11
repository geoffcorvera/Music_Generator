import os, sys
from music21 import converter, instrument, note, chord, stream
# from musicgen import parseMidi, extract_notes, export_midi

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


# Accepts a music21 stream object as input.
# Returns mapping from timestep, t, to notes/chords played at time t.
def extract_notes(s):
    allparts = s.flat
    notes = dict()

    # TODO: compare offset, if same, add to list
    for i in allparts:
        t = i.offset
        d = i.duration.quarterLength
        if t not in notes:
            notes[t] = []

        if isinstance(i, note.Note):
            p = str(i.pitch)
            notes[t].append((p,d))
        elif isinstance(i, chord.Chord):
            ps = '.'.join(str(n.pitch) for n in i.notes)
            notes[t].append((ps,d))

    return notes

# Write midi file from list notes & chords
def export_midi(notes, filename=None):
    s = stream.Stream()

    for t in notes:
        # Get list of notes played at time t
        ns = notes[t]
        for n in ns:
            if '.' in n[0]:
                ps = [pitch for pitch in n[0].split('.')]
                ch = chord.Chord(ps, quarterLength=n[1])
                s.insert(t, ch)
            else:
                nt = note.Note(n[0], quarterLength=n[1])
                s.insert(t, nt)

    if not filename:
        try:
            o_file = sys.argv[1]
            filepath = 'output/' + o_file + '.mid'
            path = s.write('midi', fp=filepath)
        except IndexError:
            path = s.write('midi', fp='output/test.mid')
    else:
        path = s.write('midi', fp=f'output/{str(filename)}.mid')

    print(f"Song written to {path}")
    

s = parseMidi('data/test', 'DEB_CLAI.mid')
notes = extract_notes(s)
export_midi(notes)
