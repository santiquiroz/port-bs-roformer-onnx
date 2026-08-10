"""Export a RoFormer checkpoint to a spec-in / mask-out ONNX graph.

    python toolkit/export_roformer.py mel_band_roformer_kim [--chunk 352800]

Writes artifacts/<name>_T<frames>.onnx and updates artifacts/manifest.json, then
gates net-level parity (torch vs ORT CPU-EP) on random spectra.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.catalog import MODELS
from toolkit.msst_loader import MSST_ROOT, build_model, load_config
from toolkit.spec_models import (
    SpecOnlyBSRoformer,
    SpecOnlyMelBandRoformer,
    patch_for_export,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
CHECKPOINTS = REPO / "checkpoints"
OPSET = 17
# Net-level gate on uniform-random spectra, which are far out of distribution
# for these nets and therefore harsher than real audio. Measured: ~5e-06.
PARITY_GATE = 1e-4

WRAPPERS = {
    "bs_roformer": SpecOnlyBSRoformer,
    "mel_band_roformer": SpecOnlyMelBandRoformer,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def make_spec(model: torch.nn.Module, audio: torch.Tensor) -> torch.Tensor:
    """[b, S, N] audio -> [b, F*S, T, 2], exactly the reference's layout."""
    from einops import rearrange

    b, s, _ = audio.shape
    window = model.stft_window_fn()
    spec = torch.stft(
        audio.reshape(b * s, -1), **model.stft_kwargs, window=window, return_complex=True
    )
    spec = torch.view_as_real(spec).reshape(b, s, *torch.view_as_real(spec).shape[1:])
    return rearrange(spec, "b s f t c -> b (f s) t c")


def export(name: str, chunk: int | None, gate: float) -> dict:
    spec_model = MODELS[name]
    config = load_config(spec_model.config)
    chunk_size = chunk or int(config.audio.chunk_size)
    checkpoint = CHECKPOINTS / spec_model.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint {checkpoint} -- download it from\n  {spec_model.checkpoint_url}")
    found = sha256(checkpoint)
    if found != spec_model.checkpoint_sha256:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch\n  expected {spec_model.checkpoint_sha256}\n  found    {found}")

    model = build_model(spec_model.arch, config, checkpoint)
    patched = patch_for_export(model)
    print(f"{name}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, patched={patched}")
    wrapper = WRAPPERS[spec_model.arch](model).eval()

    rng = np.random.default_rng(0)
    audio = torch.from_numpy((rng.standard_normal((1, 2, chunk_size)) * 0.1).astype(np.float32))
    spec = make_spec(model, audio)
    frames = spec.shape[2]
    print(f"  spec {tuple(spec.shape)}  ({chunk_size} samples, {frames} frames)")

    onnx_path = ARTIFACTS / f"{name}_T{frames}.onnx"
    ARTIFACTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper, (spec,), str(onnx_path), export_params=True, opset_version=OPSET,
        do_constant_folding=True, input_names=["spec"], output_names=["mask"], dynamo=False,
    )
    print(f"  exported in {time.perf_counter()-t0:.1f}s -> {onnx_path.stat().st_size/1e6:.1f} MB")

    import onnxruntime as ort

    with torch.inference_mode():
        ref = wrapper(spec).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"spec": spec.numpy()})[0]
    diff = np.abs(out - ref)
    max_abs = float(diff.max())
    status = "OK" if max_abs < gate else "FAIL"
    print(f"  net parity torch vs ORT-CPU: max={max_abs:.3e} rms={np.sqrt((diff**2).mean()):.3e} [{status}]")
    if status == "FAIL":
        raise SystemExit(f"{name}: net parity gate {gate} failed ({max_abs:.3e})")

    return {
        "file": onnx_path.name,
        "architecture": spec_model.arch,
        "stems": list(spec_model.stems),
        "chunk_size": chunk_size,
        "frames": frames,
        "graph_io": {
            "input": {"spec": list(spec.shape)},
            "output": {"mask": list(out.shape)},
        },
        "n_fft": int(config.model.stft_n_fft),
        "hop_length": int(config.model.stft_hop_length),
        "sample_rate": int(config.audio.sample_rate),
        "num_overlap": int(config.inference.num_overlap),
        "source_checkpoint": spec_model.checkpoint,
        "source_checkpoint_sha256": spec_model.checkpoint_sha256,
        "source_checkpoint_url": spec_model.checkpoint_url,
        "weights_author": spec_model.author,
        "weights_license": spec_model.license,
        "redistributable": spec_model.redistributable,
        "onnx_sha256": sha256(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "export_patches": patched,
        "net_parity_cpu_max_abs": max_abs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=None)
    ap.add_argument("--chunk", type=int, default=None, help="override chunk_size in samples")
    ap.add_argument("--gate", type=float, default=PARITY_GATE)
    args = ap.parse_args()

    names = args.names or list(MODELS)
    manifest_path = ARTIFACTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = manifest.get("models", {})
    for name in names:
        entries[name] = export(name, args.chunk, args.gate)
    ARTIFACTS.mkdir(exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "opset": OPSET,
                "exporter": f"legacy JIT (dynamo=False), torch {torch.__version__}",
                "msst_commit": _msst_commit(),
                "graph_boundary": "spec [1, F*S, T, 2] -> mask [1, N, F*S, T, 2]; "
                "STFT/iSTFT, the complex multiply and the DC filter are the driver's job",
                "models": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {manifest_path}")


def _msst_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(MSST_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
