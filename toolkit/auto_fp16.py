"""Find the SMALLEST set of nodes that has to stay fp32 for the mask to hold.

`export_fp16.py` blocks whole op types by hand -- RMSNorm for both architectures,
plus `ScatterElements` for the mel-band one, which has no fp16 CPU kernel with
`reduction='add'`. That is enough to stop the mask coming out NaN, but on
`mel_band_roformer_kim` it still costs 1.25 dB, so somewhere in the graph a
specific node is losing the precision that matters.

Hand-blocking more op types is guessing. This bisects instead: onnxconverter's
mixed-precision search converts, runs the graph, checks the output against the
fp32 reference and keeps narrowing which nodes must stay fp32 until the tolerance
holds. It runs on the DirectML EP on purpose -- the CPU EP has no fp16 kernel for
several of these ops, so a CPU search would blame precision for what is really a
missing kernel.

Tolerance is deliberately looser than `validate_ort.py`'s mask gate (p99.9 < 1e-4,
rms < 1e-5): those thresholds encode fp32 parity, which no fp16 graph can reach
by construction. What matters for a precision variant is the `quality` gate, so
this aims one order of magnitude below the observed fp16 error and then hands the
result to the real gates for the verdict.

    python toolkit/auto_fp16.py mel_band_roformer_kim
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
GOLDEN = REPO / "refs" / "golden"
OUT_DIR = ARTIFACTS / "fp16"

RMS_TARGET = 1e-4
P999_TARGET = 1e-3


def source_graph(name: str) -> Path:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    if name not in manifest["models"]:
        raise SystemExit(f"{name} is not in artifacts/manifest.json")
    return ARTIFACTS / manifest["models"][name]["file"]


def close_enough(reference: list[np.ndarray], candidate: list[np.ndarray]) -> bool:
    for ref, cand in zip(reference, candidate):
        diff = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
        rms = float(np.sqrt(np.mean(diff**2)))
        p999 = float(np.percentile(diff, 99.9))
        if rms > RMS_TARGET or p999 > P999_TARGET:
            return False
    return True


def main() -> None:
    from onnxconverter_common import auto_mixed_precision_model_path as amp

    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    args = ap.parse_args()

    src = source_graph(args.name)
    out = OUT_DIR / f"{src.stem}_fp16_auto.onnx"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # onnx.save appends to an existing external-data file instead of truncating it.
    out.unlink(missing_ok=True)
    (out.parent / (out.name + ".data")).unlink(missing_ok=True)

    spec = np.load(GOLDEN / args.name / "chunk0_spec.npy")
    started = time.perf_counter()
    amp.auto_convert_mixed_precision_model_path(
        str(src),
        {"spec": spec},
        str(out),
        ["DmlExecutionProvider"],
        location=out.name + ".data",
        customized_validate_func=close_enough,
        keep_io_types=True,
        verbose=True,
    )
    fp32_bytes = src.stat().st_size
    fp16_bytes = out.stat().st_size + (out.parent / (out.name + ".data")).stat().st_size
    print(f"{args.name}: searched in {(time.perf_counter() - started)/60:.1f} min")
    print(f"  {fp32_bytes/2**20:.1f} MiB -> {fp16_bytes/2**20:.1f} MiB "
          f"({100 * (1 - fp16_bytes / fp32_bytes):.1f}% smaller)")
    print(f"  {out}")


if __name__ == "__main__":
    main()
