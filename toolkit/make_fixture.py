"""Build the committed parity fixture: a synthetic stereo "song", 44.1 kHz.

Real music cannot be committed to a public repo, so the fixture is generated
from a fixed seed: a chord bed + bass + percussive transients (the
"instrumental") plus a vibrato formant tone (the "vocal"). It is not meant to
sound good -- it is meant to be deterministic, stereo, broadband and to give the
separator something with clearly distinct spectral content to act on.

The instrumental is synthesised from three independent sources, and each one is
written out on its own (`fixture_other`, `fixture_bass`, `fixture_drums`) so a
4-stem model can be scored per stem. They sum EXACTLY back to `fixture_inst`
(the scale factors are powers of two, so the split is bit-identical), which is
why `fixture_mix.wav` is unchanged by their existence and the golden dumps
captured against it stay valid.

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
# Both the chord/bass bed and the hats are synthesised hot and scaled down by
# the same exact power of two, which is what makes the per-source split exact.
SOURCE_GAIN = 0.25


def _adsr(t: np.ndarray, onsets: np.ndarray, decay: float) -> np.ndarray:
    env = np.zeros_like(t)
    for onset in onsets:
        idx = t >= onset
        env[idx] += np.exp(-(t[idx] - onset) / decay)
    return env


def chords(t: np.ndarray) -> np.ndarray:
    """The harmonic bed -- what a 4-stem model calls `other`."""
    chord = [110.0, 164.81, 220.0, 329.63]  # A2 E3 A3 E4
    env = _adsr(t, np.arange(0, t[-1], 0.5), 0.35)
    left = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(chord))
    right = sum(np.sin(2 * np.pi * f * 1.003 * t + 0.4) / (i + 1) for i, f in enumerate(chord))
    return np.stack([left * env, right * env]) * SOURCE_GAIN


def bass(t: np.ndarray) -> np.ndarray:
    """A centred 55 Hz note per bar."""
    mono = 0.6 * np.sin(2 * np.pi * 55.0 * t) * _adsr(t, np.arange(0, t[-1], 1.0), 0.5)
    return np.stack([mono, mono]) * SOURCE_GAIN


def drums(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hats: filtered noise bursts on the offbeats, wide in the stereo field."""
    noise = rng.standard_normal((2, t.size))
    return noise * _adsr(t, np.arange(0.25, t[-1], 0.5), 0.03) * 0.35 * SOURCE_GAIN


def vocal(t: np.ndarray) -> np.ndarray:
    phrases = _adsr(t, np.array([0.5, 2.0, 3.7, 5.1, 6.8, 8.4, 10.0]), 0.9)
    f0 = 196.0 * (1 + 0.02 * np.sin(2 * np.pi * 5.5 * t))  # vibrato
    phase = 2 * np.pi * np.cumsum(f0) / SR
    harmonics = sum(np.sin(k * phase) / k**1.4 for k in range(1, 12))
    formant = 1 + 0.6 * np.sin(2 * np.pi * 2.3 * t)
    mono = harmonics * phrases * formant * 0.22
    return np.stack([mono, mono * 0.95])


def build() -> dict[str, np.ndarray]:
    """Every committed fixture track, already at its final relative level."""
    rng = np.random.default_rng(SEED)
    t = np.arange(int(SR * SECONDS)) / SR
    # rng is consumed only by `drums`; keep it the first call so the noise is
    # the same sequence it has always been and the mix does not move.
    sources = {"drums": drums(t, rng), "other": chords(t), "bass": bass(t)}
    inst = sources["other"] + sources["bass"] + sources["drums"]
    voc = vocal(t)
    return {**sources, "vocal": voc, "inst": inst, "mix": inst + voc}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = build()
    scale = 0.89 / np.abs(tracks["mix"]).max()
    for name, audio in tracks.items():
        path = OUT / f"fixture_{name}.wav"
        sf.write(path, (audio * scale).T.astype(np.float32), SR, subtype="FLOAT")
        print(f"wrote {path}  {audio.shape[1]/SR:.2f}s  peak={np.abs(audio*scale).max():.3f}")


if __name__ == "__main__":
    main()
