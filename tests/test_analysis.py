import numpy as np

import vocal_pitch.analysis as analysis
from vocal_pitch.models import WordPitch, WordTiming


def test_word_pitch_rows_rounding() -> None:
    rows = analysis.word_pitch_rows(
        [
            WordPitch(
                word="hello",
                start_s=0.12345,
                end_s=0.67891,
                median_hz=440.123,
                mean_hz=441.987,
                midi_note=69.25,
                note_name="A4",
                voiced_ratio=0.87654,
            )
        ]
    )
    assert rows == [
        {
            "word": "hello",
            "start_s": 0.123,
            "end_s": 0.679,
            "median_hz": 440.12,
            "mean_hz": 441.99,
            "midi_note": 69.25,
            "note_name": "A4",
            "voiced_ratio": 0.877,
        }
    ]


def test_extract_word_pitches_with_optional_vocal_separation(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_separate(path: str, **kwargs):
        calls["separated_from"] = path
        calls["separation_kwargs"] = kwargs
        return "vocals.wav"

    monkeypatch.setattr(analysis, "separate_vocals_with_demucs", fake_separate)
    monkeypatch.setattr(
        analysis,
        "load_audio_mono",
        lambda path, sample_rate: (np.zeros(64, dtype=np.float32), sample_rate),
    )
    monkeypatch.setattr(analysis, "estimate_pitch_contour", lambda _wave, _sr: [])
    monkeypatch.setattr(
        analysis,
        "summarize_pitch_for_window",
        lambda _c, start_s, end_s: (440.0, 440.5, 1.0),
    )
    monkeypatch.setattr(analysis, "hz_to_midi_note", lambda _hz: 69.0)
    monkeypatch.setattr(analysis, "midi_to_note_name", lambda _midi: "A4")

    result = analysis.extract_word_pitches(
        "mixed.mp3",
        words=[WordTiming(word="你", start_s=0.0, end_s=0.5)],
        separate_vocals=True,
        separation_model="htdemucs_ft",
    )

    assert calls["separated_from"] == "mixed.mp3"
    assert calls["separation_kwargs"] == {
        "model_name": "htdemucs_ft",
        "output_dir": None,
        "overwrite": False,
    }
    assert len(result) == 1
    assert result[0].note_name == "A4"
