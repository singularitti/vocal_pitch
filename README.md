## vocal-pitch

Function-first project for extracting per-word sung pitch from solo vocal audio.
Supports common formats such as `mp3`, `m4a`, and `wav`.

## Install

```bash
uv sync --extra transcribe --extra dev
```

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
)
```

`df` columns (default exploded format) include:

- `token_index`, `token`
- `token_start_s`, `token_end_s`
- `note_index`
- `note_start_s`, `note_end_s`
- `note_name`, `midi_note`
- `median_hz`, `mean_hz`
- `frame_count`

Each row contains:

- `token`
- `start_s`, `end_s`
- `note_count`
- `notes` (list of note names)
- `midi_notes` (list)
- `median_hz` (list)
