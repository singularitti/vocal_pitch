from __future__ import annotations

from pathlib import Path

import av
import librosa
import numpy as np
import soundfile as sf


def _resample_if_needed(waveform: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return waveform.astype(np.float32, copy=False), sr
    resampled = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    return np.asarray(resampled, dtype=np.float32), target_sr


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    return np.asarray(mono, dtype=np.float32), int(sr)


def _normalize_pcm(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        max_abs = max(abs(info.min), info.max)
        return arr.astype(np.float32) / float(max_abs)
    return arr.astype(np.float32, copy=False)


def _load_with_av(path: Path) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None

    with av.open(str(path)) as container:
        try:
            stream = next(s for s in container.streams if s.type == "audio")
        except StopIteration as exc:
            raise RuntimeError(f"No audio stream found in {path}") from exc

        for frame in container.decode(stream):
            raw = _normalize_pcm(frame.to_ndarray())
            if raw.ndim == 1:
                mono = raw
            elif raw.ndim == 2:
                # PyAV commonly returns (channels, samples).
                mono = raw.mean(axis=0)
            else:
                mono = raw.reshape(-1)
            chunks.append(np.asarray(mono, dtype=np.float32))
            if frame.sample_rate is not None:
                sample_rate = int(frame.sample_rate)

        if sample_rate is None:
            stream_rate = stream.rate or stream.codec_context.sample_rate
            if stream_rate is None:
                raise RuntimeError(f"Cannot determine sample rate for {path}")
            sample_rate = int(stream_rate)

    if not chunks:
        raise RuntimeError(f"No decodable audio frames in {path}")
    return np.concatenate(chunks).astype(np.float32, copy=False), sample_rate


def load_audio_mono(audio_path: str | Path, sample_rate: int = 22_050) -> tuple[np.ndarray, int]:
    """
    Load audio as mono float waveform.

    The loader tries soundfile first, then PyAV fallback for formats like m4a.
    """
    path = Path(audio_path)
    try:
        waveform, sr = _load_with_soundfile(path)
    except (OSError, RuntimeError, sf.LibsndfileError):
        waveform, sr = _load_with_av(path)
    return _resample_if_needed(waveform, sr, sample_rate)
