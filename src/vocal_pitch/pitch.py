from __future__ import annotations

import math
from collections.abc import Sequence

import librosa
import numpy as np

from vocal_pitch.models import PitchFrame

_NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_midi_note(frequency_hz: float | None) -> float | None:
    if frequency_hz is None or frequency_hz <= 0:
        return None
    return 69.0 + 12.0 * math.log2(frequency_hz / 440.0)


def midi_to_note_name(midi_note: float | None) -> str | None:
    if midi_note is None:
        return None
    nearest = int(round(midi_note))
    name = _NOTE_NAMES_SHARP[nearest % 12]
    octave = nearest // 12 - 1
    return f"{name}{octave}"


def estimate_pitch_contour(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> list[PitchFrame]:
    """
    Estimate F0 contour with pYIN.

    `frequency_hz` is `None` on unvoiced frames.
    """
    if fmin_hz is None:
        fmin_hz = librosa.note_to_hz("C2")
    if fmax_hz is None:
        fmax_hz = librosa.note_to_hz("C7")

    f0_hz, _, voiced_probability = librosa.pyin(
        y=waveform,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    times_s = librosa.times_like(f0_hz, sr=sample_rate, hop_length=hop_length)
    contour: list[PitchFrame] = []
    for time_s, f0, prob in zip(times_s, f0_hz, voiced_probability, strict=False):
        hz = float(f0) if np.isfinite(f0) else None
        confidence = float(prob) if np.isfinite(prob) else None
        contour.append(PitchFrame(time_s=float(time_s), frequency_hz=hz, confidence=confidence))
    return contour


def summarize_pitch_for_window(
    contour: Sequence[PitchFrame],
    *,
    start_s: float,
    end_s: float,
) -> tuple[float | None, float | None, float]:
    """
    Return (median_hz, mean_hz, voiced_ratio) for frames in [start_s, end_s].
    """
    if end_s <= start_s:
        return None, None, 0.0

    in_window = [f for f in contour if start_s <= f.time_s <= end_s]
    if not in_window:
        return None, None, 0.0

    voiced = [f.frequency_hz for f in in_window if f.frequency_hz is not None]
    voiced_ratio = len(voiced) / len(in_window)
    if not voiced:
        return None, None, voiced_ratio

    voiced_arr = np.asarray(voiced, dtype=float)
    return float(np.median(voiced_arr)), float(np.mean(voiced_arr)), voiced_ratio
