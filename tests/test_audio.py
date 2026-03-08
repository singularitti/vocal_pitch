import numpy as np
from pathlib import Path

import vocal_pitch.audio as audio


def test_load_audio_mono_uses_soundfile_first(monkeypatch) -> None:
    stereo = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

    def fake_read(*_args, **_kwargs):
        return stereo, 22_050

    monkeypatch.setattr(audio.sf, "read", fake_read)
    monkeypatch.setattr(audio, "_load_with_av", lambda _path: (_ for _ in ()).throw(RuntimeError("unused")))

    waveform, sr = audio.load_audio_mono("dummy.wav", sample_rate=22_050)
    assert sr == 22_050
    assert np.allclose(waveform, np.asarray([2.0, 3.0], dtype=np.float32))


def test_load_audio_mono_falls_back_to_av(monkeypatch) -> None:
    def fake_read(*_args, **_kwargs):
        raise RuntimeError("soundfile failed")

    def fake_av(_path):
        return np.linspace(-1.0, 1.0, 441, dtype=np.float32), 44_100

    monkeypatch.setattr(audio.sf, "read", fake_read)
    monkeypatch.setattr(audio, "_load_with_av", fake_av)

    waveform, sr = audio.load_audio_mono("dummy.m4a", sample_rate=22_050)
    assert sr == 22_050
    assert waveform.dtype == np.float32
    assert len(waveform) > 0
    assert len(waveform) != 441


def test_normalize_pcm_int16() -> None:
    pcm = np.asarray([-32768, 0, 32767], dtype=np.int16)
    normalized = audio._normalize_pcm(pcm)
    assert normalized.dtype == np.float32
    assert np.isclose(normalized[1], 0.0)
    assert np.isclose(normalized[2], 32767 / 32768, atol=1e-5)


def test_slice_audio_mono_extracts_time_window(monkeypatch) -> None:
    waveform = np.linspace(-1.0, 1.0, 100, dtype=np.float32)

    monkeypatch.setattr(audio, "load_audio_mono", lambda _path, sample_rate: (waveform, sample_rate))

    clip, sr, start_s, end_s = audio.slice_audio_mono(
        "dummy.wav",
        start_s=0.2,
        end_s=0.4,
        sample_rate=100,
        pad_s=0.05,
    )

    assert sr == 100
    assert np.allclose(clip, waveform[15:45])
    assert start_s == 0.15
    assert end_s == 0.45


def test_play_waveform_uses_cli_backend(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class TempFile:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            Path(self.name).write_bytes(b"")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/afplay" if name == "afplay" else None)
    monkeypatch.setattr(
        audio.tempfile,
        "NamedTemporaryFile",
        lambda suffix, delete: TempFile(str(tmp_path / "clip.wav")),
    )
    monkeypatch.setattr(audio.sf, "write", lambda path, waveform, sr: calls.update(path=Path(path), sr=sr))
    monkeypatch.setattr(audio.subprocess, "run", lambda cmd, check: calls.update(cmd=cmd, check=check))

    result = audio.play_waveform(np.asarray([0.1, 0.2], dtype=np.float32), 16_000)

    assert result is None
    assert calls["sr"] == 16_000
    assert calls["cmd"] == ["afplay", str(tmp_path / "clip.wav")]
    assert calls["check"] is True
    assert not (tmp_path / "clip.wav").exists()
