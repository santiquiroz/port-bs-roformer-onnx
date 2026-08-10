"""Parity gates: exported graph + numpy driver vs the torch golden dumps.

Stages, reported separately so a regression can be attributed to one of them:

  stft    driver STFT of chunk 0             vs golden chunk0_spec
  mask    graph fed the GOLDEN spec          vs golden chunk0_mask     GATED
  synth   driver tail fed golden spec+mask   vs golden chunk0_audio    GATED
  quality driver stems and golden stems, both scored against the fixture's
          ground-truth stem                                            GATED
  drift   driver stems vs golden stems                                 informational

Why `drift` is not a gate: this architecture is chaotically sensitive to
float32-level differences in its input spectrogram. Measured on the upstream
TORCH model, with no ONNX anywhere: feeding it a float64-accurate STFT instead
of its own float32 one -- a 1.5e-05 difference on a spectrum peaking at 161 --
moves the output mask by 0.266. The same graph fed the same spectrum agrees to
1.7e-05. So sample-level agreement with a specific reference run is not
achievable by any reimplementation, and the honest question is whether the port
separates as well as the reference, which is what `quality` measures.

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

from driver.chunking import iter_chunks, pad_mix, plan_chunks
from driver.pipeline import RoformerDriver, RoformerSpec, apply_zero_dc, complex_multiply
from driver.stft import istft_bands_last, stft_bands_last
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
GATE_SYNTH_SISDR_DB = 100.0
# The driver's separation quality may not fall more than this below the
# reference's, measured against the fixture's ground-truth stem.
GATE_QUALITY_MARGIN_DB = 0.5

EPS = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider"}


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.ravel().astype(np.float64)
    est = estimate.ravel().astype(np.float64)
    alpha = float(np.dot(est, ref) / max(np.dot(ref, ref), 1e-20))
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10(np.dot(target, target) / max(np.dot(noise, noise), 1e-20)))


def _synthesis_only(spec: np.ndarray, mask: np.ndarray, cfg: RoformerSpec) -> np.ndarray:
    """The driver's tail (complex multiply -> DC filter -> iSTFT) on golden inputs."""
    masked = complex_multiply(spec[:, None], mask)[0]
    if cfg.zero_dc:
        masked = apply_zero_dc(masked, cfg.audio_channels)
    return np.stack([
        istft_bands_last(masked[n], cfg.audio_channels, cfg.n_fft, cfg.hop_length, cfg.chunk_size)
        for n in range(masked.shape[0])
    ])


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

    # 1. driver STFT vs the reference's torch.stft. Chunk 0 comes from the
    #    BORDER-PADDED mix, exactly as the reference's demix loop sees it.
    plan = plan_chunks(mix.shape[1], driver_spec.chunk_size, driver_spec.num_overlap)
    _, _, chunk0, _ = next(iter(iter_chunks(pad_mix(mix, plan), plan)))
    driver_spectrum = stft_bands_last(
        chunk0.astype(np.float32), driver_spec.n_fft, driver_spec.hop_length
    )
    st = stats(driver_spectrum - golden_spec)
    print(f"  stft   max={st['max']:.3e} rms={st['rms']:.3e} p99.9={st['p999']:.3e}")

    # 2. graph alone, fed the GOLDEN spec so driver drift cannot leak in
    mask = run(golden_spec)
    st = stats(mask - golden_mask)
    gate_ok = st["p999"] < GATE_MASK_P999 and st["rms"] < GATE_MASK_RMS
    ok &= gate_ok
    print(f"  mask   max={st['max']:.3e} rms={st['rms']:.3e} p99.9={st['p999']:.3e} "
          f"[{'OK' if gate_ok else 'FAIL'}]")

    # 3. synthesis floor: golden spec + GOLDEN mask through the driver's tail.
    #    Whatever this loses is the numpy iSTFT alone, with the net taken out.
    synth = _synthesis_only(golden_spec, golden_mask, driver_spec)
    golden_chunk_audio = np.load(golden / "chunk0_audio.npy")
    floor = si_sdr(golden_chunk_audio, synth)
    gate_ok = floor > GATE_SYNTH_SISDR_DB
    ok &= gate_ok
    print(f"  synth  chunk SI-SDR={floor:6.1f} dB [{'OK' if gate_ok else 'FAIL'}]")

    # 4. separation quality against ground truth, driver vs reference
    stems = RoformerDriver(run, driver_spec).separate(mix)
    for i, stem in enumerate(spec_model.stems):
        truth_path = FIXTURE.parent / f"fixture_{'vocal' if stem == 'vocals' else stem}.wav"
        drift = si_sdr(golden_stems[i], stems[i])
        if truth_path.exists():
            truth, _ = sf.read(truth_path, dtype="float32", always_2d=True)
            truth = np.ascontiguousarray(truth.T)
            ref_q = si_sdr(truth, golden_stems[i])
            drv_q = si_sdr(truth, stems[i])
            gate_ok = drv_q > ref_q - GATE_QUALITY_MARGIN_DB
            ok &= gate_ok
            print(f"  quality {stem:<8} reference={ref_q:6.2f} dB  driver={drv_q:6.2f} dB  "
                  f"delta={drv_q-ref_q:+.2f} dB [{'OK' if gate_ok else 'FAIL'}]")
        print(f"  drift  {stem:<9} driver vs reference SI-SDR={drift:6.1f} dB (informational)")
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
