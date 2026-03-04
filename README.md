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
