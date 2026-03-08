from __future__ import annotations

import shutil
import subprocess
import tempfile
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


def slice_audio_mono(
    audio_path: str | Path,
    *,
    start_s: float,
    end_s: float,
    sample_rate: int = 22_050,
    pad_s: float = 0.0,
) -> tuple[np.ndarray, int, float, float]:
    """
    Load a mono audio clip from `audio_path` for the requested time window.

    Returns `(waveform, sample_rate, clip_start_s, clip_end_s)`.
    """
    if start_s < 0:
        raise ValueError("`start_s` must be >= 0.")
    if end_s < start_s:
        raise ValueError("`end_s` must be >= `start_s`.")
    if pad_s < 0:
        raise ValueError("`pad_s` must be >= 0.")

    waveform, sr = load_audio_mono(audio_path, sample_rate=sample_rate)
    total_samples = len(waveform)
    if total_samples == 0:
        return np.zeros(1, dtype=np.float32), sr, 0.0, 0.0

    total_duration_s = total_samples / sr
    clip_start_s = max(0.0, start_s - pad_s)
    clip_end_s = min(total_duration_s, end_s + pad_s)

    # Ensure silent/instantaneous windows still produce a playable sample.
    if clip_end_s <= clip_start_s:
        sample_width_s = 1.0 / sr
        clip_end_s = min(total_duration_s, clip_start_s + sample_width_s)
        if clip_end_s <= clip_start_s:
            clip_start_s = max(0.0, total_duration_s - sample_width_s)
            clip_end_s = total_duration_s

    start_idx = min(int(np.floor(clip_start_s * sr)), total_samples - 1)
    end_idx = min(total_samples, max(start_idx + 1, int(np.ceil(clip_end_s * sr))))
    clip = np.asarray(waveform[start_idx:end_idx], dtype=np.float32).copy()
    return clip, sr, start_idx / sr, end_idx / sr


def play_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    backend: str = "auto",
    autoplay: bool = True,
) -> object | None:
    """
    Play a mono waveform through a local player or return an IPython audio object.
    """
    if backend == "auto":
        for candidate in ("afplay", "ffplay", "mpv", "aplay"):
            if shutil.which(candidate) is not None:
                backend = candidate
                break
        else:
            backend = "display"

    if backend in {"afplay", "ffplay", "mpv", "aplay"}:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            sf.write(temp_path, waveform, sample_rate)
            command_map = {
                "afplay": ["afplay", str(temp_path)],
                "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(temp_path)],
                "mpv": ["mpv", "--no-video", "--really-quiet", str(temp_path)],
                "aplay": ["aplay", str(temp_path)],
            }
            subprocess.run(command_map[backend], check=True)
        finally:
            temp_path.unlink(missing_ok=True)
        return None

    if backend == "display":
        try:
            from IPython.display import Audio, display
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "No CLI audio player was found and IPython is unavailable. "
                "Install IPython or play the clip manually."
            ) from exc
        audio = Audio(waveform, rate=sample_rate, autoplay=autoplay)
        display(audio)
        return audio

    raise ValueError(
        "`backend` must be one of 'auto', 'display', 'afplay', 'ffplay', 'mpv', or 'aplay'."
    )
