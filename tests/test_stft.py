"""The driver's numpy STFT/iSTFT must be interchangeable with torch's."""
from __future__ import annotations

import numpy as np
import pytest

from driver.stft import (
    hann_periodic,
    istft,
    istft_bands_last,
    stft,
    stft_bands_last,
)

torch = pytest.importorskip("torch")

N_FFT, HOP = 2048, 441


@pytest.fixture
def audio() -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(44100) / 44100
    tone = np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3300 * t)
    return np.stack([tone, tone * 0.8 + 0.05 * rng.standard_normal(t.size)]).astype(np.float32)


def test_hann_window_matches_torch():
    """Same formula; the residual is torch evaluating `1 - cos` in float32.

    Max absolute difference measured at 2.1e-07 -- under two float32 ulps at the
    window's peak. The driver keeps float64, which is the more accurate side.
    """
    ours = hann_periodic(N_FFT)
    theirs = torch.hann_window(N_FFT, periodic=True).numpy()
    assert np.abs(ours - theirs).max() < 1e-6


def test_stft_matches_torch(audio):
    ours = stft(audio, N_FFT, HOP)
    theirs = torch.stft(
        torch.from_numpy(audio), n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=torch.hann_window(N_FFT), normalized=False, return_complex=True,
    ).numpy()
    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() < 1e-3


def test_istft_matches_torch(audio):
    spec = stft(audio, N_FFT, HOP)
    length = audio.shape[1]
    ours = np.stack([istft(spec[c], N_FFT, HOP, length) for c in range(2)])
    theirs = torch.istft(
        torch.from_numpy(spec), n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=torch.hann_window(N_FFT), normalized=False, length=length,
    ).numpy()
    assert np.abs(ours - theirs).max() < 1e-4


def test_stft_istft_roundtrip(audio):
    spec = stft(audio, N_FFT, HOP)
    back = np.stack([istft(spec[c], N_FFT, HOP, audio.shape[1]) for c in range(2)])
    assert np.abs(back - audio).max() < 1e-4


def test_bands_last_layout_roundtrip(audio):
    packed = stft_bands_last(audio, N_FFT, HOP)
    assert packed.shape == (1, (N_FFT // 2 + 1) * 2, audio.shape[1] // HOP + 1, 2)
    back = istft_bands_last(packed[0], 2, N_FFT, HOP, audio.shape[1])
    assert np.abs(back - audio).max() < 1e-4


def test_bands_last_matches_reference_interleave(audio):
    """Channel is the fastest-varying axis: row 2*f+c is (freq f, channel c)."""
    packed = stft_bands_last(audio, N_FFT, HOP)[0]
    spec = stft(audio, N_FFT, HOP)
    for f in (0, 7, 1024):
        for c in (0, 1):
            row = packed[2 * f + c]
            assert np.abs(row[:, 0] - spec[c, f].real).max() < 1e-3
            assert np.abs(row[:, 1] - spec[c, f].imag).max() < 1e-3
