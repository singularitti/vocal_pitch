from vocal_pitch.analysis import word_pitch_rows
from vocal_pitch.models import WordPitch


def test_word_pitch_rows_rounding() -> None:
    rows = word_pitch_rows(
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
