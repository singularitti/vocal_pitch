## vocal-pitch

Function-first project for extracting per-word sung pitch from solo vocal audio.

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

Each row contains:

- `token`
- `start_s`, `end_s`
- `note_count`
- `notes` (list of note names)
- `midi_notes` (list)
- `median_hz` (list)
