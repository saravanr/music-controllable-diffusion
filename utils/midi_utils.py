import os.path
import numpy as np
from note_seq import play_sequence
import note_seq
from note_seq import midi_io

from note_seq.protobuf import music_pb2


def note_seq_to_nd_array(note_sequence):
    """Converts a NoteSequence serialized proto to arrays."""
    # Based on magenta/models/piano_genie/loader.py
    note_sequence_ordered = list(note_sequence.notes)

    # We do not include other header information present in NoteSeq. This may be lossy
    # but as long as we have the notes and instruments, we should have some version of the
    # song
    pitches = np.array([note.pitch for note in note_sequence_ordered])
    velocities = np.array([note.velocity for note in note_sequence_ordered])
    start_times = np.array([note.start_time for note in note_sequence_ordered])
    end_times = np.array([note.end_time for note in note_sequence_ordered])
    instruments = np.array([note.instrument for note in note_sequence_ordered])
    programs = np.array([note.program for note in note_sequence_ordered])
    nd_array = np.stack([pitches, velocities, instruments, programs, start_times, end_times], axis=1).astype(np.float32)

    # Normalize
    nd_array = (nd_array - 127.) / 254.0
    nd_array = np.tanh(nd_array)
    return nd_array


def nd_array_to_note_seq(note_array):
    """Converts ND array to Note Seq"""
    seq = music_pb2.NoteSequence()
    note_array = (np.arctanh(note_array) * 254.0) + 127.

    for i in range(0, note_array.shape[0]):
        note = music_pb2.NoteSequence.Note()
        note.pitch = int(note_array.T[0][i])
        note.velocity = int(note_array.T[1][i])
        note.instrument = int(note_array.T[2][i])
        note.program = int(note_array.T[3][i])
        note.start_time = note_array.T[4][i]
        note.end_time = note_array.T[5][i]
        seq.notes.append(note)

    return seq


def midi_file_note_seq_array(midi_file):
    midi_note = note_seq.midi_io.midi_file_to_note_sequence(midi_file)
    nd_array = note_seq_to_nd_array(midi_note)
    return nd_array


def play_midi_file(midi_file_name):
    note_sequence_object = note_seq.midi_file_to_note_sequence(midi_file_name)
    play_sequence(note_sequence_object, synth=note_seq.fluidsynth)


def get_encoding(midi_file_path):
    nd_array = midi_file_note_seq_array(midi_file_path)
    return nd_array


def save_decoder_output_as_midi(decoder_output, midi_file_name):
    seq = nd_array_to_note_seq(decoder_output)
    midi_io.note_sequence_to_midi_file(seq, midi_file_name)


if __name__ == "__main__":
    _midi_file_name = "/home/joy/data/archive/midi/lmd_full/d/d9ef4f22e5bf77cae6bda79c50887267.mid"
    _encoding = get_encoding(_midi_file_name)
    _output_file = f"/tmp/{os.path.basename(_midi_file_name)}"
    save_decoder_output_as_midi(_encoding, _output_file)
    play_midi_file(_output_file)
