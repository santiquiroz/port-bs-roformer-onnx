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
    # Resolved against the MSST checkout first, then this repo -- see
    # `msst_loader.resolve_config`. Configs published only as release assets are
    # vendored under `toolkit/configs/`.
    config: str
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
    "bs_roformer_musdb18_4stem": RoformerModel(
        name="bs_roformer_musdb18_4stem",
        arch="bs_roformer",
        # Published only as a release asset, never committed to the repo tree,
        # so it is vendored here verbatim (MIT, same repo as the weights).
        config="toolkit/configs/config_bs_roformer_384_8_2_485100.yaml",
        checkpoint="model_bs_roformer_ep_17_sdr_9.6568.ckpt",
        checkpoint_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
        checkpoint_sha256="3e9daecd70aaed5b5a0d1f861cc4d77eaa45afb3fc6301b1cf32c1be0f5868fb",
        author="Roman Solovyev (ZFTurbo)",
        # The repo's LICENSE is MIT and the weights are a release asset of that
        # same repo, which looks like an MIT grant until you read the thread
        # where the licence was added. ZFTurbo was asked, in that exact thread,
        # whether the pretrained models could be redistributed:
        #   "not all models were posted by me. While I can add some open
        #    license on repo I think I can't really decide on each model."
        #   -- ZFTurbo, MSST issue #90, 2024-11-04
        # He added the LICENSE two days later, answering a follow-up that asked
        # only about "the py code in this repo". So MIT covers the code and the
        # author explicitly declined to license the weights.
        license="NONE STATED -- the MSST repo's MIT LICENSE covers its code; its "
        "author declined to license the checkpoints (issue #90)",
        redistributable=False,
        # Order is `training.instruments` in the config, which is the order the
        # net stacks its mask estimators in. It is NOT alphabetical and it is
        # NOT the usual drums/bass/vocals/other ordering -- read it, do not
        # assume it.
        stems=("drums", "bass", "other", "vocals"),
        notes="Exports, validates and benches cleanly -- the graph is NOT published, "
        "same rule as bs_roformer_viperx_1297. Multisong avg SDR 9.38 (bass 11.08, "
        "drums 11.29, vocals 9.19, other 5.96); MUSDB18 test avg 9.65. Trained on "
        "MUSDB18HQ only (100 songs), so it is behind the dedicated vocal models on "
        "vocals -- 9.19 against Kim's 10.98 -- and the point of it is the other "
        "three stems, which no single-stem model gives. Measured here: 0.13x "
        "realtime end to end on a RX 7800 XT (see README).",
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
