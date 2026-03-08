from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchFrame:
    """Pitch estimate for one frame."""

    time_s: float
    frequency_hz: float | None
    confidence: float | None = None


@dataclass(frozen=True)
class WordTiming:
    """Word-level timing from a transcript."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class WordPitch:
    """Pitch summary for one word interval."""

    word: str
    start_s: float
    end_s: float
    median_hz: float | None
    mean_hz: float | None
    midi_note: float | None
    note_name: str | None
    voiced_ratio: float


@dataclass(frozen=True)
class NoteEvent:
    """Contiguous sung note segment."""

    start_s: float
    end_s: float
    median_hz: float
    mean_hz: float
    midi_note: float
    note_name: str
    frame_count: int


@dataclass(frozen=True)
class AudioClip:
    """Mono audio clip extracted from a time window."""

    waveform: np.ndarray
    sample_rate: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class LyricTokenNotes:
    """Mapping between one lyric token and one-or-more notes."""

    token: str
    start_s: float
    end_s: float
    notes: tuple[NoteEvent, ...]


@dataclass(frozen=True)
class LyricTokenInspection:
    """Inspectable lyric token with note metadata and preview audio."""

    token_index: int
    token: str
    start_s: float
    end_s: float
    notes: tuple[NoteEvent, ...]
    clip: AudioClip
