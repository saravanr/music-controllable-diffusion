import numpy as np
from note_seq.protobuf import music_pb2


def note_seq_to_ndarray(note_sequence):
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

    if note_sequence_ordered:
        # Delta time start high to indicate free decision
        delta_times = np.concatenate([[100000.],
                                      start_times[1:] - start_times[:-1]])
    else:
        delta_times = np.zeros_like(start_times)

    return np.stack([pitches, velocities, instruments, programs, start_times, end_times, delta_times], axis=1) \
             .astype(np.float32)


def ndarray_to_note_seq(note_array):
    """Converts ND array to Note Seq"""

    note_seq = music_pb2.NoteSequence()
    for i in range(0, note_array.shape[0]):
        note = music_pb2.NoteSequence.Note()
        note.pitch = int(note_array.T[0][i])
        note.velocity = int(note_array.T[1][i])
        note.instrument = int(note_array.T[2][i])
        note.program = int(note_array.T[3][i])
        note.start_time = note_array.T[4][i]
        note.end_time = note_array.T[5][i]
        note_seq.notes.append(note)

    return note_seq