## vocal-pitch

Function-first project for extracting per-word sung pitch from solo vocal audio.
Supports common formats such as `mp3`, `m4a`, and `wav`.

## Install

```bash
uv sync --extra transcribe --extra dev
```

For vocal/background separation, also install:

```bash
uv sync --extra separation
```

`separation` uses `demucs`, which currently requires Python `<3.14`.
This project is pinned to Python `3.13` via `.python-version`.
If Demucs reports missing `torchcodec`, run `uv sync --extra separation` again
after pulling latest `pyproject.toml`.

## Use in Python

```python
from vocal_pitch import extract_word_pitches_with_whisper, word_pitch_rows

result = extract_word_pitches_with_whisper(
    "your-song.mp3",
    model_size="small",
    language="en",
)

for row in word_pitch_rows(result):
    print(row)
```

Each row contains:

- `word`
- `start_s`, `end_s`
- `median_hz`, `mean_hz`
- `midi_note`
- `note_name` (e.g. `A4`)
- `voiced_ratio`

## Use Your Own Lyrics (No ASR Dependency For Word Boundaries)

If Whisper tokenization is not accurate enough, provide the lyrics text directly.
This mode detects note events from pitch contour and aligns tokens in order, allowing
multiple notes per token.

```python
from vocal_pitch import extract_lyrics_note_mapping, lyrics_note_rows

aligned = extract_lyrics_note_mapping(
    "your-song.mp3",
    lyrics_text="红尘醉",
)

for row in lyrics_note_rows(aligned):
    print(row)
```

Simpler one-call API:

```python
from vocal_pitch import extract_lyrics_note_rows

for row in extract_lyrics_note_rows(
    "your-song.mp3",
    lyrics_text="喔詹德為什麼沒有聲音",
):
    print(row)
```

Tokenization behavior:

- Default: `respect_spaces=False` (spaces do not change CJK char-level parsing).
- Optional: set `respect_spaces=True` if you want whitespace chunks treated as custom tokens.

If you want it to print directly:

```python
from vocal_pitch import print_lyrics_notes

print_lyrics_notes(
    "your-song.mp3",
    lyrics_text="喔 詹德 為什麼 沒有 聲音",
)
```

If you prefer a `DataFrame`:

```python
from vocal_pitch import extract_lyrics_notes_df

df = extract_lyrics_notes_df(
    "your-song.mp3",
    lyrics_text="喔 詹德 為什麼 沒有 聲音",
    explode_notes=False,
)
```

With background music, enable vocal separation first:

```python
from vocal_pitch import extract_lyrics_notes_df

df = extract_lyrics_notes_df(
    "your-song-with-piano.m4a",
    lyrics_text="喔 詹德 為什麼 沒有 聲音",
    separate_vocals=True,
    separation_model="htdemucs",
)
```

`df` columns (default exploded format) include:

- `token_index`, `token`
- `token_start_s`, `token_end_s`
- `token_duration_s`, `token_time_range`
- `note_index`
- `note_start_s`, `note_end_s`
- `note_duration_s`, `note_time_range`
- `note_name`, `midi_note`
- `median_hz`, `mean_hz`
- `frame_count`

If you want a more compact table, set `explode_notes=False`. That gives one row
per lyric token, with joined note summaries and the token time range.

If one token looks suspicious and you want to hear just that time span:

```python
from vocal_pitch import inspect_lyrics_token, play_lyrics_token

inspection = inspect_lyrics_token(
    "your-song.mp3",
    df,
    token_index=2,
    preview_pad_s=0.08,
)

print(inspection.token, inspection.start_s, inspection.end_s)
print([note.note_name for note in inspection.notes])

play_lyrics_token(
    "your-song.mp3",
    df,
    token_index=2,
)
```

Passing the existing `df` avoids recomputing alignment just to inspect one token.
`play_lyrics_token(...)` prefers a local player such as `afplay`/`ffplay`; if no
CLI player is available, it falls back to `IPython.display.Audio`.

Each row contains:

- `token`
- `start_s`, `end_s`
- `note_count`
- `notes` (list of note names)
- `midi_notes` (list)
- `median_hz` (list)
