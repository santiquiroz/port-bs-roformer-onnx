"""Throughput: one graph call, and full driver end-to-end, per execution provider.

    python toolkit/bench_dml.py mel_band_roformer_kim --ep cpu dml
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import RoformerDriver, RoformerSpec
from toolkit.catalog import MODELS
from toolkit.validate_ort import EPS, ARTIFACTS, FIXTURE, GOLDEN, graph_path, make_session


def bench_graph(sess: ort.InferenceSession, spec: np.ndarray, warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        sess.run(None, {"spec": spec})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"spec": spec})
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=None)
    ap.add_argument("--ep", nargs="+", default=["cpu", "dml"], choices=list(EPS))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--model-path", type=Path,
                    help="override the manifest graph path (requires exactly one model name)")
    args = ap.parse_args()

    names = args.names or list(MODELS)
    if args.model_path is not None and len(names) != 1:
        ap.error("--model-path requires exactly one model name")

    mix, _ = sf.read(FIXTURE, dtype="float32", always_2d=True)
    mix = np.ascontiguousarray(mix.T)

    for name in names:
        manifest = json.loads((ARTIFACTS / "manifest.json").read_text())["models"].get(name)
        if manifest is None:
            print(f"{name}: not exported, skipping")
            continue
        meta = manifest
        chunk_s = meta["chunk_size"] / meta["sample_rate"]
        spec = np.load(GOLDEN / name / "chunk0_spec.npy") if (GOLDEN / name / "chunk0_spec.npy").exists() else None
        if spec is None:
            rng = np.random.default_rng(0)
            spec = rng.standard_normal(meta["graph_io"]["input"]["spec"]).astype(np.float32)
        driver_spec = RoformerSpec(
            n_fft=meta["n_fft"], hop_length=meta["hop_length"], chunk_size=meta["chunk_size"],
            num_overlap=meta["num_overlap"], sample_rate=meta["sample_rate"],
            stems=MODELS[name].stems,
        )
        for ep in args.ep:
            if EPS[ep] not in ort.get_available_providers():
                continue
            sess = make_session(args.model_path or graph_path(name), ep)
            times = bench_graph(sess, spec, args.warmup, args.runs)
            best, med = min(times), statistics.median(times)
            line = (f"{name} [{ep}] graph: best={best*1000:8.1f} ms  median={med*1000:8.1f} ms  "
                    f"-> {chunk_s/best:6.2f}x realtime ({chunk_s:.2f}s chunk)")
            if not args.skip_e2e:
                run = lambda s: sess.run(None, {"spec": s})[0]  # noqa: E731
                t0 = time.perf_counter()
                RoformerDriver(run, driver_spec).separate(mix)
                e2e = time.perf_counter() - t0
                audio_s = mix.shape[1] / meta["sample_rate"]
                line += f"  | e2e {e2e:.2f}s for {audio_s:.1f}s ({audio_s/e2e:.2f}x)"
            print(line)


if __name__ == "__main__":
    main()
