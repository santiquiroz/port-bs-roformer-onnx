"""fp32 graph -> fp16 graph, half the download and ~12x faster on DirectML.

The TODO said fp16 was "not attempted at all" and warned that the cost in mask
accuracy was the whole job. Measured on `bs_roformer_musdb18_4stem` (RX 7800 XT,
onnxruntime-directml 1.24.4):

    size        508.0 MiB -> 254.9 MiB  (49.8% smaller)
    per chunk   17235 ms  -> 1432 ms    (12.0x)
    end to end  0.13x realtime -> 1.23x realtime
    quality     drums -0.03 dB, bass -0.06 dB, vocals -0.06 dB vs the fp32 driver

12x is far more than the ~2x that halving the arithmetic buys, and that is the
point: the fp32 attention intermediate is about 1.3 GB, so the fp32 graph spends
most of its time moving memory it cannot keep resident. Halving it is what makes
this model finish faster than realtime instead of eight times slower.

Three things have to be right or the graph will not even load:

  1. **ORT's converter, not onnxconverter_common's.** The latter leaves nodes
     whose inputs come from a converted and a non-converted producer with mixed
     dtypes, and the load fails with "Type parameter (T) of Optype (Mul) bound to
     different types" at the first attention block.
  2. **Drop the stale `value_info`.** The converter rewrites tensor types but
     leaves the old shape hints, and ORT rejects the graph on the first Cast.
  3. **Sort topologically.** The converter appends its `graph_input_cast_*` nodes
     after the nodes that consume them.

And one that has to be right or the graph loads and returns NaN: RMSNorm must
stay in fp32. `Pow` squares a spectrum that peaks around 161 and `ReduceMean`
sums those squares across the block, which passes fp16's 65504 ceiling long
before the divide brings it back down. Blocking those four ops costs 0.1 MiB.
Blocking the attention (`Einsum`, `Softmax`) on top of that changes nothing
measurable (-0.66 dB vs -0.64 dB on `other`), so it is not in the default list.

    python toolkit/export_fp16.py mel_band_roformer_kim
    python toolkit/validate_ort.py mel_band_roformer_kim --ep dml \
        --model-path artifacts/fp16/mel_band_roformer_kim_T801_fp16.onnx
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
OUT_DIR = ARTIFACTS / "fp16"

# RMSNorm. Without these four the mask comes out NaN -- see the module docstring.
DEFAULT_BLOCK = ["Pow", "ReduceMean", "Sqrt", "Div"]


def source_graph(name: str) -> Path:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    if name not in manifest["models"]:
        raise SystemExit(f"{name} is not in artifacts/manifest.json")
    return ARTIFACTS / manifest["models"][name]["file"]


def output_graph(src: Path) -> Path:
    return OUT_DIR / f"{src.stem}_fp16.onnx"


def _drop_stale_value_info(model) -> None:
    """Stale shape hints from before the conversion; ORT rejects the graph on them.

    They are intermediate hints only -- inputs and outputs keep their declared
    fp32 types -- so dropping them lets ORT infer the new types instead of
    checking the new graph against the old ones.
    """
    del model.graph.value_info[:]


def _clear_previous(out: Path) -> None:
    """onnx.save APPENDS to an existing external-data file instead of truncating.

    A second run then silently writes a graph twice the size with the same
    weights: measured 255 MiB becoming 506 MiB, and it still loads.
    """
    out.unlink(missing_ok=True)
    (out.parent / (out.name + ".data")).unlink(missing_ok=True)


def convert(name: str, block_ops: list[str]) -> Path:
    import onnx
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    src = source_graph(name)
    out = output_graph(src)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _clear_previous(out)

    started = time.perf_counter()
    # keep_io_types: the numpy driver keeps feeding and receiving fp32, so the
    # graph is the only thing that changes and the comparison stays honest.
    converted = float16.convert_float_to_float16(
        onnx.load(str(src)), keep_io_types=True, op_block_list=block_ops or None
    )
    _drop_stale_value_info(converted)
    wrapper = OnnxModel(converted)
    wrapper.topological_sort()
    converted = wrapper.model
    onnx.checker.check_model(converted, full_check=False)
    onnx.save(
        converted,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out.name + ".data",
    )

    fp32_bytes = src.stat().st_size
    fp16_bytes = out.stat().st_size + (out.parent / (out.name + ".data")).stat().st_size
    print(f"{name}: converted in {time.perf_counter() - started:.1f}s")
    print(f"  {fp32_bytes/2**20:.1f} MiB -> {fp16_bytes/2**20:.1f} MiB "
          f"({100 * (1 - fp16_bytes / fp32_bytes):.1f}% smaller)")
    print(f"  {out}")
    return out


def timed_session(graph: Path) -> None:
    import onnxruntime as ort

    # Severity 3 = errors only: fp16 constant-folding warnings for Einsum and Pow
    # flood the log and bury the real diagnostic.
    ort.set_default_logger_severity(3)
    started = time.perf_counter()
    ort.InferenceSession(str(graph), providers=["DmlExecutionProvider"])
    print(f"  DirectML session created in {time.perf_counter() - started:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="model name as it appears in artifacts/manifest.json")
    ap.add_argument("--block", default=",".join(DEFAULT_BLOCK),
                    help="comma-separated op types to keep in fp32 (empty = pure fp16)")
    ap.add_argument("--time-session", action="store_true",
                    help="also create a DirectML session and time it")
    args = ap.parse_args()

    out = convert(args.name, [op for op in args.block.split(",") if op])
    if args.time_session:
        timed_session(out)


if __name__ == "__main__":
    main()
