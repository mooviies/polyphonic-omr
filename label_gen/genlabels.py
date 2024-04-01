# Script for parsing MusicXML and generating ground truth 
# sequence labels in desired manner. This version generates ground truth
# sequence labels for the first line of the first part of the MusicXML file.
# python genlabels.py -input <.musicxmls directory> -output <.semantic directory>

import sys
import os
import argparse
from musicxml import MusicXML

if __name__ == '__main__':

    """
    Command line args:

    -input <input directory with MusicXMLS>
    -output <output directory to write to>
    """

    # Parse command line arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', type=str, required='-c' not in sys.argv, help='Path to the input directory with MusicXMLs.')
    parser.add_argument('-output', dest='output', type=str, required=True, help='Path to the output directory to write sequences.')
    parser.add_argument('-voc_p', dest='voc_p', type=str, required=True, help='Path to the pitch vocabulary file.')
    parser.add_argument('-voc_r', dest='voc_r', type=str, required=False, help='Path to the rhythm vocabulary file.')
    args = parser.parse_args()

    # Create output folders
    length_dir = os.path.join(args.output, "labels_length")
    note_dir = os.path.join(args.output, "labels_note")
    os.makedirs(length_dir, exist_ok=True)
    os.makedirs(note_dir, exist_ok=True)

    # Read vocabulary
    with open(args.voc_p, 'r') as file:
        voc_p = [element.strip() for element in file.readlines()]

    with open(args.voc_r, 'r') as file:
        voc_r = [element.strip() for element in file.readlines()]

    #print('Input dir (MusicXMLs):', args.input)
    #print('Output dir (Sequences):', args.output)

    # For tracking number of MusicXML files read
    file_num = 0

    # Go through all inputs generating output sequences
    for i, file_name in enumerate(os.listdir(args.input)):

        # Ignore non .musicxml files
        if not file_name.endswith('.musicxml'):
            continue

        # Create a MusicXML object for generating sequences
        input_path = os.path.join(args.input, file_name)
        new_file_name = ''.join(file_name.split('.')[:-1]) + '.semantic'
        output_length = os.path.join(length_dir, new_file_name)
        output_note = os.path.join(note_dir, new_file_name)
        musicxml_obj = MusicXML(input_file=input_path, output_length=output_length, output_note=output_note, voc_p=voc_p, voc_r=voc_r)

        # Generate output sequence
        try:
            musicxml_obj.write_sequences()
            file_num += 1
        except UnicodeDecodeError: # Ignore bad MusicXML
            pass

    print('Num MusicXML files read:', file_num)