from vocal_pitch.lyrics import align_tokens_to_notes, detect_note_events, tokenize_lyrics
from vocal_pitch.models import NoteEvent, PitchFrame


def test_tokenize_lyrics_cjk_without_spaces() -> None:
    assert tokenize_lyrics("红尘醉") == ["红", "尘", "醉"]


def test_detect_note_events_split_by_pitch_jump() -> None:
    contour = [
        PitchFrame(time_s=0.00, frequency_hz=440.0),
        PitchFrame(time_s=0.01, frequency_hz=442.0),
        PitchFrame(time_s=0.02, frequency_hz=439.0),
        PitchFrame(time_s=0.03, frequency_hz=523.25),
        PitchFrame(time_s=0.04, frequency_hz=525.0),
        PitchFrame(time_s=0.05, frequency_hz=523.25),
    ]
    events = detect_note_events(
        contour,
        pitch_jump_semitones=0.8,
        max_unvoiced_gap_s=0.05,
        min_note_duration_s=0.0,
    )
    assert len(events) == 2
    assert events[0].note_name == "A4"
    assert events[1].note_name == "C5"


def test_align_tokens_to_notes_one_to_many() -> None:
    notes = [
        NoteEvent(0.0, 0.1, 440.0, 440.0, 69.0, "A4", 4),
        NoteEvent(0.1, 0.2, 493.88, 493.88, 71.0, "B4", 4),
        NoteEvent(0.2, 0.3, 523.25, 523.25, 72.0, "C5", 4),
        NoteEvent(0.3, 0.4, 587.33, 587.33, 74.0, "D5", 4),
    ]
    aligned = align_tokens_to_notes(["你", "好吗"], notes)
    assert len(aligned) == 2
    assert len(aligned[0].notes) == 1
    assert [n.note_name for n in aligned[1].notes] == ["B4", "C5", "D5"]
