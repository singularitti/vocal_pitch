from vocal_pitch.models import PitchFrame
from vocal_pitch.pitch import hz_to_midi_note, midi_to_note_name, summarize_pitch_for_window


def test_hz_to_midi_note_a4() -> None:
    midi = hz_to_midi_note(440.0)
    assert midi is not None
    assert round(midi, 5) == 69.0
    assert midi_to_note_name(midi) == "A4"


def test_summarize_pitch_for_window() -> None:
    contour = [
        PitchFrame(time_s=0.0, frequency_hz=None),
        PitchFrame(time_s=0.1, frequency_hz=220.0),
        PitchFrame(time_s=0.2, frequency_hz=230.0),
        PitchFrame(time_s=0.3, frequency_hz=None),
    ]
    median_hz, mean_hz, voiced_ratio = summarize_pitch_for_window(
        contour,
        start_s=0.0,
        end_s=0.3,
    )

    assert median_hz == 225.0
    assert mean_hz == 225.0
    assert voiced_ratio == 0.5
