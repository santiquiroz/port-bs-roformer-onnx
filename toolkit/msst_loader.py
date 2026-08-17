"""Build an MSST RoFormer from its upstream config + checkpoint.

The architecture code is NOT vendored: `MSST_ROOT` points at a checkout of
ZFTurbo/Music-Source-Separation-Training (MIT), so the golden reference and the
exported graph are always the same source of truth, at a pinned commit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
MSST_ROOT = Path(os.environ.get("MSST_ROOT", REPO / "msst"))


class TupleSafeLoader(yaml.SafeLoader):
    """SafeLoader plus the one non-standard tag MSST configs use."""


TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def resolve_config(rel_path: str) -> Path:
    """A catalog `config` path, against the MSST checkout then this repo.

    Most upstream configs live in the checkout's `configs/`. A few are published
    only as GitHub release assets and never land in the repo tree; those are
    vendored under `toolkit/configs/` (they are MIT, same as the checkout) so a
    model is not unbuildable just because its config is not a tracked file.
    """
    for root in (MSST_ROOT, REPO):
        path = root / rel_path
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{rel_path} found under neither {MSST_ROOT} nor {REPO} "
        "-- set MSST_ROOT or run toolkit/setup-env.ps1"
    )


def load_config(rel_path: str):
    from ml_collections import ConfigDict

    path = resolve_config(rel_path)
    return ConfigDict(yaml.load(path.read_text(), Loader=TupleSafeLoader))


def _import_arch(arch: str):
    if str(MSST_ROOT) not in sys.path:
        sys.path.insert(0, str(MSST_ROOT))
    if arch == "bs_roformer":
        from models.bs_roformer.bs_roformer import BSRoformer

        return BSRoformer
    if arch == "mel_band_roformer":
        from models.bs_roformer.mel_band_roformer import MelBandRoformer

        return MelBandRoformer
    raise ValueError(f"unknown architecture {arch!r}")


def build_model(arch: str, config, checkpoint: Path) -> torch.nn.Module:
    cls = _import_arch(arch)
    model = cls(**dict(config.model))
    # weights_only: third-party .ckpt, never unpickle arbitrary objects.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)  # strict: an arch mismatch must fail loudly
    model.eval()
    return model
