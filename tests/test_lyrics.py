import vocal_pitch.lyrics as lyrics
import pandas as pd
from vocal_pitch.models import LyricTokenNotes, NoteEvent, PitchFrame


def test_tokenize_lyrics_cjk_without_spaces() -> None:
    assert lyrics.tokenize_lyrics("红尘醉") == ["红", "尘", "醉"]


def test_detect_note_events_split_by_pitch_jump() -> None:
    contour = [
        PitchFrame(time_s=0.00, frequency_hz=440.0),
        PitchFrame(time_s=0.01, frequency_hz=442.0),
        PitchFrame(time_s=0.02, frequency_hz=439.0),
        PitchFrame(time_s=0.03, frequency_hz=523.25),
        PitchFrame(time_s=0.04, frequency_hz=525.0),
        PitchFrame(time_s=0.05, frequency_hz=523.25),
    ]
    events = lyrics.detect_note_events(
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
    aligned = lyrics.align_tokens_to_notes(["你", "好吗"], notes)
    assert len(aligned) == 2
    assert len(aligned[0].notes) == 1
    assert [n.note_name for n in aligned[1].notes] == ["B4", "C5", "D5"]


def test_extract_lyrics_note_rows_wrapper(monkeypatch) -> None:
    fake_aligned = [
        LyricTokenNotes(token="你", start_s=0.0, end_s=0.2, notes=tuple()),
    ]

    def fake_extract(*_args, **_kwargs):
        return fake_aligned

    monkeypatch.setattr(lyrics, "extract_lyrics_note_mapping", fake_extract)
    rows = lyrics.extract_lyrics_note_rows("dummy.mp3", "你")
    assert rows == [
        {
            "token": "你",
            "start_s": 0.0,
            "end_s": 0.2,
            "note_count": 0,
            "notes": [],
            "midi_notes": [],
            "median_hz": [],
        }
    ]


def test_print_lyrics_notes_wrapper(monkeypatch, capsys) -> None:
    fake_rows = [
        {
            "token": "你",
            "start_s": 0.0,
            "end_s": 0.2,
            "note_count": 1,
            "notes": ["A4"],
            "midi_notes": [69.0],
            "median_hz": [440.0],
        }
    ]

    def fake_extract_rows(*_args, **_kwargs):
        return fake_rows

    monkeypatch.setattr(lyrics, "extract_lyrics_note_rows", fake_extract_rows)
    rows = lyrics.print_lyrics_notes("dummy.mp3", "你")
    captured = capsys.readouterr()

    assert rows == fake_rows
    assert "A4" in captured.out


def test_lyrics_note_dataframe_exploded() -> None:
    aligned = [
        LyricTokenNotes(
            token="你",
            start_s=0.0,
            end_s=0.4,
            notes=(
                NoteEvent(0.0, 0.2, 440.0, 441.0, 69.0, "A4", 8),
                NoteEvent(0.2, 0.4, 493.88, 494.0, 71.0, "B4", 8),
            ),
        ),
        LyricTokenNotes(token="好", start_s=0.4, end_s=0.6, notes=tuple()),
    ]
    df = lyrics.lyrics_note_dataframe(aligned, explode_notes=True)
    assert list(df["token"]) == ["你", "你", "好"]
    assert list(df["note_name"][:2]) == ["A4", "B4"]
    assert pd.isna(df["note_name"].iloc[2])


def test_extract_lyrics_notes_df_wrapper(monkeypatch) -> None:
    fake_aligned = [LyricTokenNotes(token="你", start_s=0.0, end_s=0.2, notes=tuple())]

    def fake_extract(*_args, **_kwargs):
        return fake_aligned

    monkeypatch.setattr(lyrics, "extract_lyrics_note_mapping", fake_extract)
    df = lyrics.extract_lyrics_notes_df("dummy.mp3", "你")
    assert list(df["token"]) == ["你"]
