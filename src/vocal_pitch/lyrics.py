from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from vocal_pitch.audio import load_audio_mono
from vocal_pitch.models import LyricTokenNotes, NoteEvent, PitchFrame
from vocal_pitch.pitch import estimate_pitch_contour, hz_to_midi_note, midi_to_note_name


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_token_delimiter(ch: str) -> bool:
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def tokenize_lyrics(lyrics_text: str) -> list[str]:
    """
    Split provided lyrics into tokens.

    Rules:
    - If the text has spaces, split by spaces.
    - For CJK without spaces, split by character.
    - Non-CJK runs stay grouped (e.g., latin words).
    """
    stripped = lyrics_text.strip()
    if not stripped:
        return []

    if any(ch.isspace() for ch in stripped):
        return [tok for tok in stripped.split() if tok]

    tokens: list[str] = []
    buffer: list[str] = []
    for ch in stripped:
        if _is_token_delimiter(ch):
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()
            continue
        if _is_cjk_char(ch):
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()
            tokens.append(ch)
            continue
        buffer.append(ch)

    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _semitone_distance(hz1: float, hz2: float) -> float:
    return abs(12.0 * math.log2(hz1 / hz2))


def _to_note_event(frames: list[PitchFrame]) -> NoteEvent | None:
    voiced_hz = [f.frequency_hz for f in frames if f.frequency_hz is not None]
    if not voiced_hz:
        return None
    values = np.asarray(voiced_hz, dtype=float)
    median_hz = float(np.median(values))
    mean_hz = float(np.mean(values))
    midi = hz_to_midi_note(median_hz)
    if midi is None:
        return None
    name = midi_to_note_name(midi)
    if name is None:
        return None
    return NoteEvent(
        start_s=float(frames[0].time_s),
        end_s=float(frames[-1].time_s),
        median_hz=median_hz,
        mean_hz=mean_hz,
        midi_note=float(midi),
        note_name=name,
        frame_count=len(voiced_hz),
    )


def detect_note_events(
    contour: Sequence[PitchFrame],
    *,
    pitch_jump_semitones: float = 0.8,
    max_unvoiced_gap_s: float = 0.05,
    min_note_duration_s: float = 0.05,
) -> list[NoteEvent]:
    """
    Split pitch contour into contiguous note events.
    """
    events: list[NoteEvent] = []
    current: list[PitchFrame] = []
    last_voiced_hz: float | None = None
    last_voiced_time: float | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        event = _to_note_event(current)
        if event is not None and (event.end_s - event.start_s) >= min_note_duration_s:
            events.append(event)
        current = []

    for frame in contour:
        hz = frame.frequency_hz
        if hz is None:
            if current and last_voiced_time is not None:
                if frame.time_s - last_voiced_time > max_unvoiced_gap_s:
                    flush()
            continue

        if not current:
            current = [frame]
            last_voiced_hz = hz
            last_voiced_time = frame.time_s
            continue

        assert last_voiced_hz is not None
        assert last_voiced_time is not None
        gap = frame.time_s - last_voiced_time
        jump = _semitone_distance(hz, last_voiced_hz)
        if gap > max_unvoiced_gap_s or jump >= pitch_jump_semitones:
            flush()
            current = [frame]
        else:
            current.append(frame)
        last_voiced_hz = hz
        last_voiced_time = frame.time_s

    flush()
    return events


def _token_weights(tokens: Sequence[str]) -> list[float]:
    weights: list[float] = []
    for token in tokens:
        weight = sum(1 for ch in token if not _is_token_delimiter(ch))
        weights.append(float(max(1, weight)))
    return weights


def align_tokens_to_notes(tokens: Sequence[str], notes: Sequence[NoteEvent]) -> list[LyricTokenNotes]:
    """
    Assign contiguous note ranges to each token.

    A token may receive multiple notes or none.
    """
    token_list = list(tokens)
    note_list = list(notes)
    m = len(token_list)
    n = len(note_list)
    if m == 0:
        return []
    if n == 0:
        return [
            LyricTokenNotes(token=token, start_s=0.0, end_s=0.0, notes=tuple())
            for token in token_list
        ]

    weights = _token_weights(token_list)
    weight_sum = sum(weights)
    expected = [n * w / weight_sum for w in weights]

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    prev = [[0] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(1, m + 1):
        for j in range(n + 1):
            best = inf
            best_p = 0
            for p in range(j + 1):
                base = dp[i - 1][p]
                if base == inf:
                    continue
                seg_len = j - p
                penalty = (seg_len - expected[i - 1]) ** 2
                if seg_len == 0:
                    penalty += 0.5
                score = base + penalty
                if score < best:
                    best = score
                    best_p = p
            dp[i][j] = best
            prev[i][j] = best_p

    ranges: list[tuple[int, int]] = [(0, 0)] * m
    j = n
    for i in range(m, 0, -1):
        p = prev[i][j]
        ranges[i - 1] = (p, j)
        j = p

    aligned: list[LyricTokenNotes] = []
    for idx, token in enumerate(token_list):
        start_idx, end_idx = ranges[idx]
        seg = tuple(note_list[start_idx:end_idx])
        if seg:
            start_s = seg[0].start_s
            end_s = seg[-1].end_s
        else:
            left_t = note_list[start_idx - 1].end_s if start_idx > 0 else note_list[0].start_s
            right_t = note_list[start_idx].start_s if start_idx < n else note_list[-1].end_s
            midpoint = (left_t + right_t) / 2.0
            start_s = midpoint
            end_s = midpoint
        aligned.append(
            LyricTokenNotes(
                token=token,
                start_s=float(start_s),
                end_s=float(end_s),
                notes=seg,
            )
        )
    return aligned


def extract_lyrics_note_mapping(
    audio_path: str | Path,
    lyrics_text: str,
    *,
    sample_rate: int = 22_050,
    pitch_jump_semitones: float = 0.8,
    max_unvoiced_gap_s: float = 0.05,
    min_note_duration_s: float = 0.05,
) -> list[LyricTokenNotes]:
    """
    Map user-provided lyrics text to detected sung note events.
    """
    tokens = tokenize_lyrics(lyrics_text)
    waveform, sr = load_audio_mono(audio_path, sample_rate=sample_rate)
    contour = estimate_pitch_contour(waveform, sr)
    note_events = detect_note_events(
        contour,
        pitch_jump_semitones=pitch_jump_semitones,
        max_unvoiced_gap_s=max_unvoiced_gap_s,
        min_note_duration_s=min_note_duration_s,
    )
    return align_tokens_to_notes(tokens, note_events)


def lyrics_note_rows(
    aligned: Sequence[LyricTokenNotes],
) -> list[dict[str, float | int | list[float] | list[str] | str]]:
    """
    Convert aligned token-note mappings into printable row dicts.
    """
    rows: list[dict[str, float | int | list[float] | list[str] | str]] = []
    for item in aligned:
        rows.append(
            {
                "token": item.token,
                "start_s": round(item.start_s, 3),
                "end_s": round(item.end_s, 3),
                "note_count": len(item.notes),
                "notes": [n.note_name for n in item.notes],
                "midi_notes": [round(n.midi_note, 2) for n in item.notes],
                "median_hz": [round(n.median_hz, 2) for n in item.notes],
            }
        )
    return rows


def extract_lyrics_note_rows(
    audio_path: str | Path,
    lyrics_text: str,
    *,
    sample_rate: int = 22_050,
    pitch_jump_semitones: float = 0.8,
    max_unvoiced_gap_s: float = 0.05,
    min_note_duration_s: float = 0.05,
) -> list[dict[str, float | int | list[float] | list[str] | str]]:
    """
    One-call helper that returns printable token-note rows.
    """
    aligned = extract_lyrics_note_mapping(
        audio_path,
        lyrics_text,
        sample_rate=sample_rate,
        pitch_jump_semitones=pitch_jump_semitones,
        max_unvoiced_gap_s=max_unvoiced_gap_s,
        min_note_duration_s=min_note_duration_s,
    )
    return lyrics_note_rows(aligned)


def print_lyrics_notes(
    audio_path: str | Path,
    lyrics_text: str,
    *,
    sample_rate: int = 22_050,
    pitch_jump_semitones: float = 0.8,
    max_unvoiced_gap_s: float = 0.05,
    min_note_duration_s: float = 0.05,
) -> list[dict[str, float | int | list[float] | list[str] | str]]:
    """
    One-call helper that computes and prints token-note rows.
    """
    rows = extract_lyrics_note_rows(
        audio_path,
        lyrics_text,
        sample_rate=sample_rate,
        pitch_jump_semitones=pitch_jump_semitones,
        max_unvoiced_gap_s=max_unvoiced_gap_s,
        min_note_duration_s=min_note_duration_s,
    )
    for row in rows:
        print(row)
    return rows


def lyrics_note_dataframe(
    aligned: Sequence[LyricTokenNotes],
    *,
    explode_notes: bool = True,
) -> pd.DataFrame:
    """
    Convert token-note mappings into a pandas DataFrame.
    """
    records: list[dict[str, str | float | int | None]] = []
    for token_index, item in enumerate(aligned):
        if explode_notes:
            if item.notes:
                for note_index, note in enumerate(item.notes):
                    records.append(
                        {
                            "token_index": token_index,
                            "token": item.token,
                            "token_start_s": round(item.start_s, 3),
                            "token_end_s": round(item.end_s, 3),
                            "note_index": note_index,
                            "note_start_s": round(note.start_s, 3),
                            "note_end_s": round(note.end_s, 3),
                            "note_name": note.note_name,
                            "midi_note": round(note.midi_note, 2),
                            "median_hz": round(note.median_hz, 2),
                            "mean_hz": round(note.mean_hz, 2),
                            "frame_count": note.frame_count,
                        }
                    )
            else:
                records.append(
                    {
                        "token_index": token_index,
                        "token": item.token,
                        "token_start_s": round(item.start_s, 3),
                        "token_end_s": round(item.end_s, 3),
                        "note_index": None,
                        "note_start_s": None,
                        "note_end_s": None,
                        "note_name": None,
                        "midi_note": None,
                        "median_hz": None,
                        "mean_hz": None,
                        "frame_count": 0,
                    }
                )
            continue

        records.append(
            {
                "token_index": token_index,
                "token": item.token,
                "token_start_s": round(item.start_s, 3),
                "token_end_s": round(item.end_s, 3),
                "note_count": len(item.notes),
                "notes": ",".join(n.note_name for n in item.notes),
                "midi_notes": ",".join(f"{n.midi_note:.2f}" for n in item.notes),
                "median_hz": ",".join(f"{n.median_hz:.2f}" for n in item.notes),
            }
        )
    return pd.DataFrame.from_records(records)


def extract_lyrics_notes_df(
    audio_path: str | Path,
    lyrics_text: str,
    *,
    sample_rate: int = 22_050,
    pitch_jump_semitones: float = 0.8,
    max_unvoiced_gap_s: float = 0.05,
    min_note_duration_s: float = 0.05,
    explode_notes: bool = True,
) -> pd.DataFrame:
    """
    One-call helper that returns a DataFrame for `audio_path` + `lyrics_text`.
    """
    aligned = extract_lyrics_note_mapping(
        audio_path,
        lyrics_text,
        sample_rate=sample_rate,
        pitch_jump_semitones=pitch_jump_semitones,
        max_unvoiced_gap_s=max_unvoiced_gap_s,
        min_note_duration_s=min_note_duration_s,
    )
    return lyrics_note_dataframe(aligned, explode_notes=explode_notes)
