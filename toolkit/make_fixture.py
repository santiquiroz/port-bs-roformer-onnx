"""Build the committed parity fixture: a synthetic stereo "song", 44.1 kHz.

Real music cannot be committed to a public repo, so the fixture is generated
from a fixed seed: a chord bed + bass + percussive transients (the
"instrumental") plus a vibrato formant tone (the "vocal"). It is not meant to
sound good -- it is meant to be deterministic, stereo, broadband and to give the
separator something with clearly distinct spectral content to act on.

    python toolkit/make_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "refs" / "inputs"
SR = 44100
SECONDS = 12.0
SEED = 20260810


def _adsr(t: np.ndarray, onsets: np.ndarray, decay: float) -> np.ndarray:
    env = np.zeros_like(t)
    for onset in onsets:
        idx = t >= onset
        env[idx] += np.exp(-(t[idx] - onset) / decay)
    return env


def instrumental(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    chord = [110.0, 164.81, 220.0, 329.63]  # A2 E3 A3 E4
    beats = np.arange(0, t[-1], 0.5)
    env = _adsr(t, beats, 0.35)
    left = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(chord))
    right = sum(np.sin(2 * np.pi * f * 1.003 * t + 0.4) / (i + 1) for i, f in enumerate(chord))
    bass = 0.6 * np.sin(2 * np.pi * 55.0 * t) * _adsr(t, np.arange(0, t[-1], 1.0), 0.5)
    # hats: filtered noise bursts on the offbeats, wide in the stereo field
    noise = rng.standard_normal((2, t.size))
    hats = noise * _adsr(t, np.arange(0.25, t[-1], 0.5), 0.03) * 0.35
    stereo = np.stack([left * env + bass, right * env + bass])
    return stereo * 0.25 + hats * 0.25


def vocal(t: np.ndarray) -> np.ndarray:
    phrases = _adsr(t, np.array([0.5, 2.0, 3.7, 5.1, 6.8, 8.4, 10.0]), 0.9)
    f0 = 196.0 * (1 + 0.02 * np.sin(2 * np.pi * 5.5 * t))  # vibrato
    phase = 2 * np.pi * np.cumsum(f0) / SR
    harmonics = sum(np.sin(k * phase) / k**1.4 for k in range(1, 12))
    formant = 1 + 0.6 * np.sin(2 * np.pi * 2.3 * t)
    mono = harmonics * phrases * formant * 0.22
    return np.stack([mono, mono * 0.95])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t = np.arange(int(SR * SECONDS)) / SR
    inst = instrumental(t, rng)
    voc = vocal(t)
    mix = inst + voc
    peak = np.abs(mix).max()
    scale = 0.89 / peak
    for name, audio in (("fixture_mix", mix), ("fixture_vocal", voc), ("fixture_inst", inst)):
        path = OUT / f"{name}.wav"
        sf.write(path, (audio * scale).T.astype(np.float32), SR, subtype="FLOAT")
        print(f"wrote {path}  {audio.shape[1]/SR:.2f}s  peak={np.abs(audio*scale).max():.3f}")


if __name__ == "__main__":
    main()
