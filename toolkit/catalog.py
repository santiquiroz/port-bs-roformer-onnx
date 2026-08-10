"""The checkpoints this port knows how to export, and what may be redistributed.

`redistributable` is the field that decides whether a graph goes into the GitHub
release or has to be built locally by the user. It is set from the LICENCE the
weights are actually published under -- not from how good the model is.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoformerModel:
    name: str
    arch: str  # "bs_roformer" | "mel_band_roformer"
    config: str  # path under msst/configs (upstream, MIT)
    checkpoint: str  # file name expected in models/
    checkpoint_url: str
    checkpoint_sha256: str
    author: str
    license: str
    redistributable: bool
    stems: tuple[str, ...]
    notes: str = ""
    extra: dict = field(default_factory=dict)


MODELS: dict[str, RoformerModel] = {
    "mel_band_roformer_kim": RoformerModel(
        name="mel_band_roformer_kim",
        arch="mel_band_roformer",
        config="configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        checkpoint="MelBandRoformer.ckpt",
        checkpoint_url="https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
        checkpoint_sha256="87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e",
        author="KimberleyJensen (KimberleyJSN)",
        license="MIT (declared on the HF model card)",
        redistributable=True,
        stems=("vocals",),
        notes="SDR vocals 10.98 on MSST's multisong benchmark -- the highest-scoring "
        "roformer vocal checkpoint that carries an explicit permissive licence.",
    ),
    "bs_roformer_viperx_1297": RoformerModel(
        name="bs_roformer_viperx_1297",
        arch="bs_roformer",
        config="configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        checkpoint="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        checkpoint_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        checkpoint_sha256="5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
        author="viperx",
        license="NONE STATED -- host repo TRvlvr/model_repo has no LICENSE file",
        redistributable=False,
        stems=("vocals",),
        notes="Supported by the toolkit so you can export it yourself. The graph is "
        "NOT published in this repo's release: no one has granted redistribution "
        "rights for these weights.",
    ),
}
