import os.path

import numpy as np
from note_seq import play_sequence
import miditoolkit
import note_seq
from muzic.musicbert.preprocess import MIDI_to_encoding, encoding_to_MIDI


def play_midi_file(midi_file_name):
    note_sequence_object = note_seq.midi_file_to_note_sequence(midi_file_name)
    play_sequence(note_sequence_object, synth=note_seq.fluidsynth)


def get_encoding(midi_file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_file_path)
    encoding = MIDI_to_encoding(midi_obj)
    nd_array = np.array(encoding)
    return nd_array


def save_as_midi(encoding, midi_file_name):
    midi_obj = encoding_to_MIDI(encoding)
    midi_obj.dump(midi_file_name)


if __name__ == "__main__":
    _midi_file_name = "/home/joy/midi/lmd_full/d/d9ef4f22e5bf77cae6bda79c50887267.mid"
    _encoding = get_encoding(_midi_file_name)
    _output_file = f"/tmp/{os.path.basename(_midi_file_name)}"
    save_as_midi(_encoding, _output_file)
    play_midi_file(_output_file)
