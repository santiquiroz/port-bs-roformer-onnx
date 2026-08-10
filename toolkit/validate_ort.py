"""Parity gates: exported graph + numpy driver vs the torch golden dumps.

Three stages, reported separately so a regression can be attributed:

  stft      driver STFT of chunk 0            vs golden chunk0_spec
  mask      graph on the GOLDEN spec          vs golden chunk0_mask   (graph alone)
  stems     full driver on the whole fixture  vs golden stems         (end to end)

    python toolkit/validate_ort.py mel_band_roformer_kim --ep cpu dml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import RoformerDriver, RoformerSpec
from driver.stft import stft_bands_last
from toolkit.catalog import MODELS

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
GOLDEN = REPO / "refs" / "golden"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"

# Gates. p99.9 + RMS bind; max-abs is printed but informational, because float32
# reassociation in a 24-layer transformer puts a handful of outliers well above
# the bulk of the distribution without moving the audio at all.
GATE_MASK_P999 = 1e-4
GATE_MASK_RMS = 1e-5
GATE_STEM_SISDR_DB = 60.0

EPS = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider"}


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.ravel().astype(np.float64)
    est = estimate.ravel().astype(np.float64)
    alpha = float(np.dot(est, ref) / max(np.dot(ref, ref), 1e-20))
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10(np.dot(target, target) / max(np.dot(noise, noise), 1e-20)))


def stats(diff: np.ndarray) -> dict:
    return {
        "max": float(np.abs(diff).max()),
        "rms": float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
        "p999": float(np.percentile(np.abs(diff), 99.9)),
    }


def make_session(path: Path, ep: str) -> ort.InferenceSession:
    providers = [("DmlExecutionProvider", {"device_id": 0})] if ep == "dml" else []
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), providers=providers)


def graph_path(name: str) -> Path:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    return ARTIFACTS / manifest["models"][name]["file"]


def validate(name: str, ep: str) -> bool:
    spec_model = MODELS[name]
    golden = GOLDEN / name
    meta = json.loads((golden / "meta.json").read_text())
    driver_spec = RoformerSpec(
        n_fft=meta["n_fft"],
        hop_length=meta["hop_length"],
        chunk_size=meta["chunk_size"],
        num_overlap=meta["num_overlap"],
        sample_rate=meta["sample_rate"],
        stems=spec_model.stems,
    )
    sess = make_session(graph_path(name), ep)
    run = lambda spec: sess.run(None, {"spec": spec})[0]  # noqa: E731

    ok = True
    golden_spec = np.load(golden / "chunk0_spec.npy")
    golden_mask = np.load(golden / "chunk0_mask.npy")
    golden_stems = np.load(golden / "stems.npy")

    mix, _ = sf.read(FIXTURE, dtype="float32", always_2d=True)
    mix = np.ascontiguousarray(mix.T)

    # 1. driver STFT vs the reference's torch.stft (on chunk 0, padded like the reference)
    chunk = mix[:, : driver_spec.chunk_size]
    if chunk.shape[1] < driver_spec.chunk_size:
        chunk = np.pad(chunk, ((0, 0), (0, driver_spec.chunk_size - chunk.shape[1])), mode="reflect")
    st = stats(stft_bands_last(chunk, driver_spec.n_fft, driver_spec.hop_length) - golden_spec)
    print(f"  stft   max={st['max']:.3e} rms={st['rms']:.3e} p99.9={st['p999']:.3e}")

    # 2. graph alone, fed the GOLDEN spec so driver drift cannot leak in
    mask = run(golden_spec)
    st = stats(mask - golden_mask)
    gate_ok = st["p999"] < GATE_MASK_P999 and st["rms"] < GATE_MASK_RMS
    ok &= gate_ok
    print(f"  mask   max={st['max']:.3e} rms={st['rms']:.3e} p99.9={st['p999']:.3e} "
          f"[{'OK' if gate_ok else 'FAIL'}]")

    # 3. end to end
    stems = RoformerDriver(run, driver_spec).separate(mix)
    for i, stem in enumerate(spec_model.stems):
        sdr = si_sdr(golden_stems[i], stems[i])
        gate_ok = sdr > GATE_STEM_SISDR_DB
        ok &= gate_ok
        d = stats(stems[i] - golden_stems[i])
        print(f"  stem {stem:<10} SI-SDR={sdr:6.1f} dB  max={d['max']:.3e} rms={d['rms']:.3e} "
              f"[{'OK' if gate_ok else 'FAIL'}]")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=None)
    ap.add_argument("--ep", nargs="+", default=["cpu", "dml"], choices=list(EPS))
    args = ap.parse_args()

    all_ok = True
    for name in args.names or list(MODELS):
        if not (GOLDEN / name / "meta.json").exists():
            print(f"{name}: no golden dump, skipping (run toolkit/capture_baseline.py)")
            continue
        for ep in args.ep:
            if EPS[ep] not in ort.get_available_providers():
                print(f"{name} [{ep}]: provider unavailable, skipping")
                continue
            print(f"{name} [{ep}]")
            all_ok &= validate(name, ep)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
