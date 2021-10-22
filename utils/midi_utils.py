import numpy as np
from note_seq.protobuf import music_pb2
import note_seq
from note_seq import midi_io
from note_seq import play_sequence


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
    Converts Midi files to Numpy arrays and vice-versa for training/eval purposes
    """

    def __init__(self):
        self._note_seq = None
        self._nd_array = None
        pass

    def load_file(self, midi_file_path):
        """
        Load from midi file
        :param midi_file_path:  The midi file path
        :return: The converter
        """
        self._note_seq = midi_io.midi_file_to_note_sequence(midi_file_path)
        return self

    def load_nd_array(self, nd_array):
        """
        Load from nd array
        :param nd_array: The nd array
        :return: The converter
        """
        self._nd_array = nd_array
        return self

    def to_nd_array(self):
        """
        Convert note sequence to nd arrays
        Based on magenta/models/piano_genie/loader.py
        :return:
        """
        if self._note_seq is None:
            raise RuntimeError(f"Please load a midi file")

        note_sequence_ordered = list(self._note_seq.notes)

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
        self._nd_array = nd_array
        return nd_array

    def to_note_seq(self):
        """
        Convert the nd array to note seq
        :return: Note seq
        """
        if self._nd_array is None:
            raise RuntimeError(f"Please load nd array")
        note_array = self._nd_array.T

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
    nd_array = MidiConverter().load_file("/home/joy/midi/lmd_full/d/d9ef4f22e5bf77cae6bda79c50887267.mid").to_nd_array()
    note_seq = MidiConverter().load_nd_array(nd_array).to_note_seq()
    play_note_seq(note_seq)
