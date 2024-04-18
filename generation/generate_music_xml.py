import os
import argparse
from pathlib import Path
from music21 import stream, note, meter, key, clef, chord
from music21.exceptions21 import TimeSignatureException
from music21.pitch import PitchException, AccidentalException

# Parse command line arguments for input
parser = argparse.ArgumentParser()
parser.add_argument('-notes', dest='notes', type=str, required=True,
                    help='Folder containing the notes .semantic files.')
parser.add_argument('-rhythms', dest='rhythms', type=str, required=True,
                    help='Folder containing the rhythms .semantic files.')
parser.add_argument('-output', dest='output', type=str, required=True,
                    help='Folder where the music xml files will be saved.')
args = parser.parse_args()

note_folder = Path(args.notes)
rhythms_folder = Path(args.rhythms)
output_folder = Path(args.output)
os.makedirs(output_folder, exist_ok=True)

for note_path in note_folder.iterdir():
    rhythm_path = rhythms_folder / note_path.name
    output_path = output_folder / note_path.name.replace('.semantic', '.musicxml')

    with open(note_path, 'r') as file:
        notes = [[element.strip() for element in line.strip().split()] for line in file.readline().split('+')]

    with open(rhythm_path, 'r') as file:
        rhythms = [[element.strip() for element in line.strip().split()] for line in file.readline().split('+')]

    sheet = stream.Score()
    sheet.append(clef.TrebleClef() if "clef-G" in notes[0][0] else clef.BassClef())
    key_signature = notes[1][0].strip('-')[1].replace('M', '')

    try:
        sheet.append(key.Key(key_signature))
    except PitchException:
        ...

    measure = stream.Part()
    for i in range(2, len(notes)):
        accord_notes = notes[i]

        if i < len(rhythms):
            accord_rhythms = rhythms[i]
        else:
            accord_rhythms = ['note-quarter']

        if len(accord_notes) == 0:
            continue

        if i == 2 and "timeSignature" in accord_notes[0]:
            try:
                measure.append(meter.TimeSignature(accord_notes[0].split('-')[1]))
            except TimeSignatureException:
                ...
            del accord_notes[0]
            del accord_rhythms[0]

        if "barline" in accord_notes:
            accord_notes.remove("barline")

        if "barline" in accord_rhythms:
            accord_rhythms.remove("barline")

        n = None
        if len(accord_notes) == 1:
            note_str = accord_notes[0]
            if "rest" in note_str:
                n = note.Rest()
                try:
                    n.duration.type = note_str.split('-')[1]
                except ValueError:
                    ...
                except IndexError:
                    ...
            elif "note" in note_str:
                try:
                    n = note.Note(note_str.split('-')[1])
                    try:
                        n.duration.type = accord_rhythms[0].split('-')[1]
                    except ValueError:
                        ...
                    except IndexError:
                        ...
                except AccidentalException:
                    ...
        else:
            for element in accord_notes:
                try:
                    element.split('-')[1]
                except IndexError:
                    print(element)

            for element in accord_rhythms:
                try:
                    element.split('-')[1]
                except IndexError:
                    print(element)

            try:
                chord_notes = [element.split('-')[1] for element in accord_notes]
                i = 0
                while i < len(chord_notes):
                    if "/" in chord_notes[i]:
                        del chord_notes[i]
                    elif len(chord_notes[i]) > 4:
                        del chord_notes[i]
                    else:
                        i += 1

                try:
                    n = chord.Chord(chord_notes)
                except AccidentalException:
                    print(chord_notes)
            except PitchException:
                print([element.split('-')[1] for element in accord_notes])

            if n is not None:
                try:
                    n.duration.type = accord_rhythms[0].split('-')[1]
                except ValueError:
                    try:
                        n.duration.type = accord_rhythms[1].split('-')[1]
                    except ValueError:
                        ...
                    except IndexError:
                        ...
                except IndexError:
                    ...

                measure.append(n)

    sheet.append(measure)
    sheet.write('musicxml', output_path)
