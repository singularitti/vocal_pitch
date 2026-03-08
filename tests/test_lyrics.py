import vocal_pitch.lyrics as lyrics
import pandas as pd
import numpy as np
import pytest
from vocal_pitch.models import AudioClip, LyricTokenInspection, LyricTokenNotes, NoteEvent, PitchFrame


def test_tokenize_lyrics_cjk_without_spaces() -> None:
    assert lyrics.tokenize_lyrics("红尘醉") == ["红", "尘", "醉"]


def test_tokenize_lyrics_ignores_spaces_by_default() -> None:
    assert lyrics.tokenize_lyrics("红尘 醉") == ["红", "尘", "醉"]


def test_tokenize_lyrics_respects_spaces_when_requested() -> None:
    assert lyrics.tokenize_lyrics("红尘 醉", respect_spaces=True) == ["红尘", "醉"]


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


def test_detect_note_events_filters_low_confidence_noise() -> None:
    contour = [
        PitchFrame(time_s=0.00, frequency_hz=440.0, confidence=0.15),
        PitchFrame(time_s=0.01, frequency_hz=441.0, confidence=0.2),
        PitchFrame(time_s=0.02, frequency_hz=None, confidence=None),
        PitchFrame(time_s=0.03, frequency_hz=523.25, confidence=0.95),
        PitchFrame(time_s=0.04, frequency_hz=523.0, confidence=0.92),
    ]
    events = lyrics.detect_note_events(
        contour,
        pitch_jump_semitones=0.8,
        max_unvoiced_gap_s=0.05,
        min_note_duration_s=0.0,
        min_note_confidence=0.6,
    )
    assert len(events) == 1
    assert events[0].note_name == "C5"


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


def test_align_tokens_to_notes_prefers_duration_over_raw_note_count() -> None:
    notes = [
        NoteEvent(0.0, 0.02, 440.0, 440.0, 69.0, "A4", 2),
        NoteEvent(0.02, 0.04, 466.16, 466.16, 70.0, "A#4", 2),
        NoteEvent(0.04, 0.5, 493.88, 493.88, 71.0, "B4", 10),
    ]

    aligned = lyrics.align_tokens_to_notes(["你", "好"], notes)

    assert [note.note_name for note in aligned[0].notes] == ["A4", "A#4"]
    assert [note.note_name for note in aligned[1].notes] == ["B4"]


def test_distribute_empty_token_windows_spreads_gap() -> None:
    windows = lyrics._distribute_empty_token_windows(0.2, 0.4, 2)

    assert windows[0] == pytest.approx((0.2, 0.3))
    assert windows[1] == pytest.approx((0.3, 0.4))


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
    assert list(df["token_time_range"][:2]) == ["0.000-0.400", "0.000-0.400"]
    assert list(df["note_time_range"][:2]) == ["0.000-0.200", "0.200-0.400"]


def test_lyrics_note_dataframe_compact_has_time_summary() -> None:
    aligned = [
        LyricTokenNotes(
            token="你",
            start_s=0.1,
            end_s=0.35,
            notes=(NoteEvent(0.1, 0.35, 440.0, 441.0, 69.0, "A4", 8),),
        )
    ]

    df = lyrics.lyrics_note_dataframe(aligned, explode_notes=False)

    assert list(df["token_duration_s"]) == [0.25]
    assert list(df["token_time_range"]) == ["0.100-0.350"]
    assert list(df["notes"]) == ["A4"]


def test_extract_lyrics_notes_df_wrapper(monkeypatch) -> None:
    fake_aligned = [LyricTokenNotes(token="你", start_s=0.0, end_s=0.2, notes=tuple())]

    def fake_extract(*_args, **_kwargs):
        return fake_aligned

    monkeypatch.setattr(lyrics, "extract_lyrics_note_mapping", fake_extract)
    df = lyrics.extract_lyrics_notes_df("dummy.mp3", "你")
    assert list(df["token"]) == ["你"]


def test_extract_lyrics_mapping_with_optional_vocal_separation(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_separate(path: str, **kwargs):
        calls["separated_from"] = path
        calls["separation_kwargs"] = kwargs
        return "vocals.wav"

    monkeypatch.setattr(lyrics, "separate_vocals_with_demucs", fake_separate)
    monkeypatch.setattr(
        lyrics,
        "load_audio_mono",
        lambda path, sample_rate: (np.zeros(32, dtype=np.float32), sample_rate),
    )
    monkeypatch.setattr(lyrics, "estimate_pitch_contour", lambda _wave, _sr: [])
    monkeypatch.setattr(lyrics, "detect_note_events", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        lyrics,
        "align_tokens_to_notes",
        lambda tokens, _notes: [
            LyricTokenNotes(token=t, start_s=0.0, end_s=0.0, notes=tuple()) for t in tokens
        ],
    )

    aligned = lyrics.extract_lyrics_note_mapping(
        "mixed.mp3",
        "你好",
        separate_vocals=True,
        separation_model="htdemucs_ft",
    )
    assert [x.token for x in aligned] == ["你", "好"]
    assert calls["separated_from"] == "mixed.mp3"
    assert calls["separation_kwargs"] == {
        "model_name": "htdemucs_ft",
        "output_dir": None,
        "overwrite": False,
    }


def test_inspect_lyrics_token_returns_clip(monkeypatch) -> None:
    fake_aligned = [
        LyricTokenNotes(
            token="你",
            start_s=0.4,
            end_s=0.5,
            notes=(NoteEvent(0.4, 0.5, 440.0, 440.0, 69.0, "A4", 4),),
        ),
        LyricTokenNotes(token="好", start_s=0.5, end_s=0.8, notes=tuple()),
    ]
    calls: dict[str, object] = {}

    def fake_extract(*_args, **_kwargs):
        return fake_aligned

    def fake_slice(*_args, **kwargs):
        calls["slice_kwargs"] = kwargs
        return np.asarray([0.1, 0.2], dtype=np.float32), 22_050, 0.275, 0.625

    monkeypatch.setattr(lyrics, "extract_lyrics_note_mapping", fake_extract)
    monkeypatch.setattr(lyrics, "slice_audio_mono", fake_slice)

    inspection = lyrics.inspect_lyrics_token("dummy.mp3", "你好", 0)

    assert inspection.token_index == 0
    assert inspection.token == "你"
    assert inspection.start_s == 0.4
    assert inspection.end_s == 0.5
    assert inspection.notes == fake_aligned[0].notes
    assert inspection.clip.sample_rate == 22_050
    assert inspection.clip.start_s == 0.275
    assert inspection.clip.end_s == 0.625
    assert np.allclose(inspection.clip.waveform, np.asarray([0.1, 0.2], dtype=np.float32))
    assert calls["slice_kwargs"] == {
        "start_s": 0.325,
        "end_s": 0.575,
        "sample_rate": 22_050,
        "pad_s": 0.05,
    }


def test_inspect_lyrics_token_uses_dataframe_without_recomputing(monkeypatch) -> None:
    df = pd.DataFrame(
        [
            {
                "token_index": 0,
                "token": "你",
                "token_start_s": 0.4,
                "token_end_s": 0.5,
                "note_index": 0,
                "note_start_s": 0.4,
                "note_end_s": 0.5,
                "note_name": "A4",
                "midi_note": 69.0,
                "median_hz": 440.0,
                "mean_hz": 440.5,
                "frame_count": 4,
            }
        ]
    )
    monkeypatch.setattr(
        lyrics,
        "extract_lyrics_note_mapping",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not recompute")),
    )
    monkeypatch.setattr(
        lyrics,
        "slice_audio_mono",
        lambda *_args, **_kwargs: (np.asarray([0.1], dtype=np.float32), 22_050, 0.35, 0.55),
    )

    inspection = lyrics.inspect_lyrics_token("dummy.mp3", df, 0)

    assert inspection.token == "你"
    assert inspection.notes[0].note_name == "A4"
    assert inspection.token_index == 0


def test_play_lyrics_token_uses_audio_backend(monkeypatch) -> None:
    fake_inspection = LyricTokenInspection(
        token_index=0,
        token="你",
        start_s=0.0,
        end_s=0.2,
        notes=tuple(),
        clip=AudioClip(
            waveform=np.asarray([0.1, 0.2], dtype=np.float32),
            sample_rate=16_000,
            start_s=0.0,
            end_s=0.2,
        ),
    )

    monkeypatch.setattr(lyrics, "inspect_lyrics_token", lambda *_args, **_kwargs: fake_inspection)
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        lyrics,
        "play_waveform",
        lambda waveform, sample_rate, *, backend, autoplay: calls.update(
            waveform=waveform,
            sample_rate=sample_rate,
            backend=backend,
            autoplay=autoplay,
        )
        or "played",
    )

    audio_obj = lyrics.play_lyrics_token("dummy.mp3", "你", 0, backend="afplay", autoplay=True)

    assert audio_obj == "played"
    assert np.allclose(calls["waveform"], np.asarray([0.1, 0.2], dtype=np.float32))
    assert calls["sample_rate"] == 16_000
    assert calls["backend"] == "afplay"
    assert calls["autoplay"] is True
