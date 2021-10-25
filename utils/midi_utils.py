import numpy as np
from note_seq.protobuf import music_pb2
import note_seq
from note_seq import midi_io
from note_seq import play_sequence
import psutil

def save_as_midi_file(note_sequence_object, output_file_path):
    """
    Save the note sequence to Midi file
    :param note_sequence_object: The note seq
    :param output_file_path: The file path
    :return:
    """
    midi_io.note_sequence_to_midi_file(note_sequence_object, output_file_path)


def play_note_seq(note_sequence_object):
    """
    Play the note sequence
    :param note_sequence_object: The note sequence object
    :return:
    """
    play_sequence(note_sequence_object, synth=note_seq.fluidsynth)


class MidiConverter(object):
    """
    Converts Midi files to Numpy arrays and vice-versa for training/eval purposes. Default max note size is 10010 (multiple
    of 7)
    since the percentiles of notes lengths are:
    [p25, p50,   p75,   p95,    p99]
    [4.0, 234.0, 863.0, 3263.0, 9636.359999999986]
    """
    def __init__(self):
        pass

    @staticmethod
    def load_file(midi_file_path):
        """
        Load from midi file
        :param midi_file_path:  The midi file path
        :return: The converter
        """
        # print(f"{psutil.virtual_memory()} -> Loading {midi_file_path}")
        sequence = midi_io.midi_file_to_note_sequence(midi_file_path)
        nd_array = MidiConverter.to_nd_array(sequence)
        return nd_array, sequence

    @staticmethod
    def to_nd_array(sequence, max_notes=10010):
        """
        Convert note sequence to nd arrays
        Based on magenta/models/piano_genie/loader.py
        :return:
        """
        if sequence is None:
            raise RuntimeError(f"Please load a midi file")

        note_sequence_ordered = list(sequence.notes)
        if len(note_sequence_ordered) > max_notes:
            # Trim
            note_sequence_ordered = note_sequence_ordered[0:max_notes]

        # We do not include other header information present in NoteSeq. This may be lossy
        # but as long as we have the notes and instruments, we should have some version of the
        # song
        pitches = np.array([note.pitch for note in note_sequence_ordered])
        velocities = np.array([note.velocity for note in note_sequence_ordered])
        start_times = np.array([note.start_time for note in note_sequence_ordered])
        end_times = np.array([note.end_time for note in note_sequence_ordered])
        instruments = np.array([note.instrument for note in note_sequence_ordered])
        programs = np.array([note.program for note in note_sequence_ordered])

        if note_sequence_ordered:
            # Delta time start high to indicate free decision
            delta_times = np.concatenate([[100000.], start_times[1:] - start_times[:-1]])
        else:
            delta_times = np.zeros_like(start_times)

        nd_array = np.stack([pitches, velocities, instruments, programs, start_times, end_times, delta_times], axis=1) \
            .astype(np.float32)
        diff_rows = max_notes - nd_array.shape[0]
        if diff_rows > 0:
            nd_array = np.pad(nd_array, ((0, diff_rows), (0, 0)), constant_values=0)
        return nd_array

    @staticmethod
    def to_note_seq(note_array):
        """
        Convert the nd array to note seq
        :return: Note seq
        """
        if note_array is None:
            raise RuntimeError(f"Please load nd array")

        note_seq_object = music_pb2.NoteSequence()
        for i in range(0, note_array.shape[1]):
            note = music_pb2.NoteSequence.Note()
            note.pitch = int(note_array[0][i])
            note.velocity = int(note_array[1][i])
            note.instrument = int(note_array[2][i])
            note.program = int(note_array[3][i])
            note.start_time = note_array[4][i]
            note.end_time = note_array[5][i]
            note_seq_object.notes.append(note)

        return note_seq_object


if __name__ == "__main__":
    _nd_array, _sequence = MidiConverter().load_file(
        "/home/joy/midi/lmd_full/d/d9ef4f22e5bf77cae6bda79c50887267.mid")
    _note_seq = MidiConverter().to_note_seq(_nd_array)
    play_note_seq(_note_seq)
