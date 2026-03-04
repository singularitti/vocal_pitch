from __future__ import annotations

from pathlib import Path

from vocal_pitch.models import WordTiming


def _normalize_word(word: str) -> str:
    stripped = word.strip()
    return stripped.strip(".,!?;:\"'()[]{}")


def transcribe_words_with_faster_whisper(
    audio_path: str | Path,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    vad_filter: bool = False,
) -> list[WordTiming]:
    """
    Transcribe word timings from audio.

    Requires optional dependency: `faster-whisper`.
    """
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "faster-whisper is not installed. Install with: `uv add faster-whisper` "
            "or `uv sync --extra transcribe`."
        ) from exc

    model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(Path(audio_path)),
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
    )

    words: list[WordTiming] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None) or []
        for word in segment_words:
            if word.start is None or word.end is None or word.word is None:
                continue
            token = _normalize_word(word.word)
            if not token:
                continue
            words.append(WordTiming(word=token, start_s=float(word.start), end_s=float(word.end)))

    return words
