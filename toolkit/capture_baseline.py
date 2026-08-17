"""Golden reference: run the model through MSST's own torch path on the fixture.

Everything the driver later has to reproduce is dumped here -- the per-chunk
spectrum, the per-chunk mask, and the final overlap-added stem -- so parity can
be attributed to a stage instead of guessed at.

    python toolkit/capture_baseline.py mel_band_roformer_kim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.catalog import MODELS
from toolkit.export_roformer import CHECKPOINTS, make_spec
from toolkit.msst_loader import build_model, load_config

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"
GOLDEN = REPO / "refs" / "golden"


def windowing_array(chunk_size: int, fade_size: int) -> np.ndarray:
    window = np.ones(chunk_size)
    window[-fade_size:] = np.linspace(1, 0, fade_size)
    window[:fade_size] = np.linspace(0, 1, fade_size)
    return window


def demix_reference(
    model, mix: np.ndarray, chunk_size: int, num_overlap: int, num_stems: int, dumps: dict
):
    """MSST `demix` generic mode, batch_size=1, in torch -- the golden path."""
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    tensor = torch.from_numpy(mix).float()
    if length_init > 2 * border and border > 0:
        tensor = torch.nn.functional.pad(tensor, (border, border), mode="reflect")
    total = tensor.shape[1]
    result = torch.zeros((num_stems, *tensor.shape))
    counter = torch.zeros((num_stems, *tensor.shape))
    base_window = torch.from_numpy(windowing_array(chunk_size, fade_size)).float()

    with torch.inference_mode():
        for index, start in enumerate(range(0, total, step)):
            part = tensor[:, start : start + chunk_size]
            seg_len = part.shape[-1]
            pad_mode = "reflect" if seg_len > chunk_size // 2 else "constant"
            part = torch.nn.functional.pad(part, (0, chunk_size - seg_len), mode=pad_mode)
            out = model(part[None])
            if out.ndim == 3:  # mel_band collapses the stem axis when num_stems == 1
                out = out[:, None]
            window = base_window.clone()
            if start == 0:
                window[:fade_size] = 1
            elif start + step >= total:
                window[-fade_size:] = 1
            if index == 0:
                dumps["chunk0_input"] = part.numpy()
                dumps["chunk0_output"] = out[0].numpy()
            result[..., start : start + seg_len] += out[0, ..., :seg_len] * window[:seg_len]
            counter[..., start : start + seg_len] += window[:seg_len]

    estimated = (result / counter).numpy()
    np.nan_to_num(estimated, copy=False, nan=0.0)
    if length_init > 2 * border and border > 0:
        estimated = estimated[..., border:-border]
    return estimated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=None)
    args = ap.parse_args()

    mix, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    mix = np.ascontiguousarray(mix.T)
    for name in args.names or list(MODELS):
        spec_model = MODELS[name]
        config = load_config(spec_model.config)
        # UNPATCHED on purpose: the golden is upstream MSST as-is (flash SDPA,
        # nn.GLU), so the parity numbers include whatever the export patches cost.
        model = build_model(spec_model.arch, config, CHECKPOINTS / spec_model.checkpoint)
        # The catalog names the stems; the net decides how many there are. A
        # disagreement means the entry is wrong about the checkpoint, and every
        # number downstream would be attributed to the wrong instrument.
        num_stems = len(model.mask_estimators)
        if num_stems != len(spec_model.stems):
            raise SystemExit(
                f"{name}: catalog declares {len(spec_model.stems)} stems "
                f"{spec_model.stems} but the checkpoint has {num_stems} mask estimators"
            )
        out_dir = GOLDEN / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # spec/mask of chunk 0, through the same wrapper the graph was traced from
        from toolkit.export_roformer import WRAPPERS

        wrapper = WRAPPERS[spec_model.arch](model).eval()
        dumps: dict[str, np.ndarray] = {}
        stems = demix_reference(
            model,
            mix,
            int(config.audio.chunk_size),
            int(config.inference.num_overlap),
            num_stems,
            dumps,
        )
        with torch.inference_mode():
            chunk_spec = make_spec(model, torch.from_numpy(dumps["chunk0_input"])[None])
            chunk_mask = wrapper(chunk_spec).numpy()
        np.save(out_dir / "chunk0_spec.npy", chunk_spec.numpy())
        np.save(out_dir / "chunk0_mask.npy", chunk_mask)
        np.save(out_dir / "chunk0_audio.npy", dumps["chunk0_output"])
        np.save(out_dir / "stems.npy", stems)
        for i, stem in enumerate(spec_model.stems):
            sf.write(out_dir / f"{stem}.wav", stems[i].T.astype(np.float32), sr, subtype="FLOAT")
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "architecture": spec_model.arch,
                    "stems": list(spec_model.stems),
                    "chunk_size": int(config.audio.chunk_size),
                    "num_overlap": int(config.inference.num_overlap),
                    "n_fft": int(config.model.stft_n_fft),
                    "hop_length": int(config.model.stft_hop_length),
                    "sample_rate": sr,
                    "fixture_samples": int(mix.shape[1]),
                    "golden_is_unpatched_upstream": True,
                    "torch": torch.__version__,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"{name}: golden written to {out_dir} (stems {stems.shape})")


if __name__ == "__main__":
    main()
