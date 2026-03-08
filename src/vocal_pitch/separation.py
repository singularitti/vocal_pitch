from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


def _demucs_output_path(output_dir: Path, model_name: str, audio_path: Path) -> Path:
    return output_dir / model_name / audio_path.stem / "vocals.wav"


def separate_vocals_with_demucs(
    audio_path: str | Path,
    *,
    model_name: str = "htdemucs",
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Separate vocals stem from mixed audio using Demucs.

    Returns the generated `vocals.wav` path.
    """
    if importlib.util.find_spec("demucs") is None:
        raise ModuleNotFoundError(
            "Demucs is not installed. Install with: `uv add demucs` or "
            "`uv sync --extra separation`."
        )

    source_path = Path(audio_path).expanduser().resolve()
    out_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path(tempfile.gettempdir()) / "vocal_pitch_demucs")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    vocals_path = _demucs_output_path(out_root, model_name, source_path)
    if vocals_path.exists() and not overwrite:
        return vocals_path

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        model_name,
        "-o",
        str(out_root),
        str(source_path),
    ]
    subprocess.run(command, check=True)

    if vocals_path.exists():
        return vocals_path

    matches = list((out_root / model_name).glob("**/vocals.wav"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        "Demucs finished but `vocals.wav` was not found at the expected location."
    )
