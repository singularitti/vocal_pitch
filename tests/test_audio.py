import numpy as np

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
