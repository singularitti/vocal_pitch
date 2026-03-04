from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio_mono(audio_path: str | Path, sample_rate: int = 22_050) -> tuple[np.ndarray, int]:
    """
    Load audio as mono float waveform.

    The loader accepts MP3 and other formats supported by librosa/soundfile.
    """
    path = Path(audio_path)
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sr
