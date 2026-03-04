from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from vocal_pitch.audio import load_audio_mono
from vocal_pitch.models import WordPitch, WordTiming
from vocal_pitch.pitch import (
    estimate_pitch_contour,
    hz_to_midi_note,
    midi_to_note_name,
    summarize_pitch_for_window,
)
from vocal_pitch.transcript import transcribe_words_with_faster_whisper

WordTranscriber = Callable[[str | Path], Sequence[WordTiming]]


def extract_word_pitches(
    audio_path: str | Path,
    *,
    words: Sequence[WordTiming] | None = None,
    transcriber: WordTranscriber | None = None,
    sample_rate: int = 22_050,
) -> list[WordPitch]:
    """
    Extract pitch summary for each word in a vocal recording.

    If `words` is omitted, provide `transcriber` to create word timings first.
    """
    if words is None:
        if transcriber is None:
            raise ValueError("Either `words` or `transcriber` must be provided.")
        words = transcriber(audio_path)

    waveform, sr = load_audio_mono(audio_path, sample_rate=sample_rate)
    contour = estimate_pitch_contour(waveform, sr)

    results: list[WordPitch] = []
    for word in words:
        median_hz, mean_hz, voiced_ratio = summarize_pitch_for_window(
            contour,
            start_s=word.start_s,
            end_s=word.end_s,
        )
        midi = hz_to_midi_note(median_hz)
        results.append(
            WordPitch(
                word=word.word,
                start_s=word.start_s,
                end_s=word.end_s,
                median_hz=median_hz,
                mean_hz=mean_hz,
                midi_note=midi,
                note_name=midi_to_note_name(midi),
                voiced_ratio=voiced_ratio,
            )
        )
    return results


def extract_word_pitches_with_whisper(
    audio_path: str | Path,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    vad_filter: bool = False,
    sample_rate: int = 22_050,
) -> list[WordPitch]:
    """
    End-to-end helper: word transcription + pitch extraction.
    """

    def _transcriber(path: str | Path) -> list[WordTiming]:
        return transcribe_words_with_faster_whisper(
            path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_filter=vad_filter,
        )

    return extract_word_pitches(audio_path, transcriber=_transcriber, sample_rate=sample_rate)


def word_pitch_rows(word_pitches: Sequence[WordPitch]) -> list[dict[str, float | str | None]]:
    """
    Represent results in a print/table-friendly row format.
    """
    return [
        {
            "word": wp.word,
            "start_s": round(wp.start_s, 3),
            "end_s": round(wp.end_s, 3),
            "median_hz": round(wp.median_hz, 2) if wp.median_hz is not None else None,
            "mean_hz": round(wp.mean_hz, 2) if wp.mean_hz is not None else None,
            "midi_note": round(wp.midi_note, 2) if wp.midi_note is not None else None,
            "note_name": wp.note_name,
            "voiced_ratio": round(wp.voiced_ratio, 3),
        }
        for wp in word_pitches
    ]
