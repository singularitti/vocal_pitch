from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import vocal_pitch.separation as separation


def test_separate_vocals_requires_demucs(monkeypatch) -> None:
    monkeypatch.setattr(separation.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ModuleNotFoundError):
        separation.separate_vocals_with_demucs("input.mp3")


def test_separate_vocals_uses_cached_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(separation.importlib.util, "find_spec", lambda _name: object())
    audio_path = tmp_path / "song.mp3"
    out_dir = tmp_path / "separated"
    cached = out_dir / "htdemucs" / "song" / "vocals.wav"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"ok")

    monkeypatch.setattr(
        separation.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run demucs")),
    )

    result = separation.separate_vocals_with_demucs(audio_path, output_dir=out_dir)
    assert result == cached


def test_separate_vocals_runs_demucs_and_returns_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(separation.importlib.util, "find_spec", lambda _name: object())

    audio_path = tmp_path / "mix.m4a"
    audio_path.write_bytes(b"fake")
    out_dir = tmp_path / "stems"

    def fake_run(cmd: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert "demucs.separate" in cmd
        out_idx = cmd.index("-o")
        model_idx = cmd.index("-n")
        root = Path(cmd[out_idx + 1])
        model_name = cmd[model_idx + 1]
        stem = Path(cmd[-1]).stem
        vocals = root / model_name / stem / "vocals.wav"
        vocals.parent.mkdir(parents=True, exist_ok=True)
        vocals.write_bytes(b"vocals")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(separation.subprocess, "run", fake_run)

    result = separation.separate_vocals_with_demucs(audio_path, output_dir=out_dir)
    assert result == out_dir / "htdemucs" / "mix" / "vocals.wav"
    assert result.exists()
