# TODO

State as of the first public commit. Everything listed under "Done" is measured, not assumed;
everything below it is genuinely not done.

## Done

- Export path for both architectures (`bs_roformer`, `mel_band_roformer`), spec-in / mask-out,
  opset 17, legacy JIT exporter. Mel filterbank gather + overlap-averaging scatter are baked
  into the graph as constants, so the driver is architecture-agnostic.
- Torch-free numpy driver: STFT, complex multiply, DC filter, iSTFT, overlapping-chunk
  overlap-add. Enforced torch-free by a test that parses the imports and by a subprocess run
  with `torch` poisoned.
- Staged parity gates against unpatched upstream MSST torch, on CPU EP and DirectML EP.
  All three catalog models are validated the same way, `bs_roformer_viperx_1297` included
  (2026-08-17) — no model rests on a short-chunk spot check any more.
- DirectML GLU miscomputation found, characterised with micro-graphs, and worked around.
- `mel_band_roformer_kim` (MIT weights) exported, validated, benchmarked and published in the
  `models-v1.0` release with `manifest.json` (source + graph SHA-256).
- **Multi-stem.** `bs_roformer_musdb18_4stem` (drums/bass/other/vocals) exported, golden
  captured, all gates green on CPU and DirectML, benchmarked. The driver needed no change; the
  toolkit did — `capture_baseline.py` hardcoded `num_stems = 1` and the fixture had no
  per-instrument ground truth. Both fixed, and the fixture now emits its three instrumental
  sources separately with `fixture_mix.wav` unchanged sample-for-sample, so the existing golden
  stays valid. The graph is **not** published: see the licence finding in the README.

### Upflow integration (done)

Integrated in Upflow v0.62.0. `RoformerSeparator` builds on `OnnxStemSeparator`, so it reuses the
session cache, cancellation and 44.1 kHz I/O. The emitted stem is paired with a residual made by
plain subtraction, without an MDX `compensate` factor. Long inputs run in overlapping chunks with
crossfade, and the audio UI warns before selecting this deliberately slow model. End-to-end it
takes about 50 seconds per minute of audio on the RX 7800 XT, roughly 20x slower than MDX. That is
the relevant product cost, not the earlier 9x graph-to-graph comparison: RoFormer processes chunks
with 50% overlap.

## Not done

### fp16 graph
**Measured 2026-08-17 — see `docs/fp16-findings.md`. Nothing fp16 is published, and the
reason is not the same for the two models.**

`bs_roformer_musdb18_4stem`: 508.0 → 254.9 MiB, 17235 → 1432 ms per chunk (**12x**), end to
end 0.13x → **1.23x realtime**, costing 0.03-0.06 dB on the three stems the model actually
separates. That flips this model from unusable to faster than realtime. It cannot be
published anyway — the weights are the licence problem below — but it means that if that
licence is ever resolved, the model ships as something usable rather than as a 31-minute
wait per song.

`mel_band_roformer_kim`: 887.9 → 444.5 MiB, 1852 → 725 ms per chunk (2.6x), end to end 1.18x
→ 2.83x realtime, costing **1.25 dB** on vocals. This is the one that could be published, and
it is the one that fails. The fp16 output agrees with the fp32 output at only 14.4 dB SI-SDR
where the 4-stem stems agree at 30-40 dB, so this is a real regression and not the weak
fixture talking. Keeping `MatMul` in fp32 — which leaves nearly every weight at fp32 and
gives up the size win — does not recover it, so there is no block list that fixes it.

Still open: an automatic mixed-precision search would answer whether some non-obvious set of
nodes recovers `kim`. `toolkit/auto_fp16.py` has the setup; the converter's own baseline
check aborts before converting anything, which is a bug in it, not a tolerance problem
(this graph is bit-deterministic on the DirectML EP — two runs agree to exactly 0.0).

### ~~BS-RoFormer golden + gates~~ — done 2026-08-17
`bs_roformer_viperx_1297` now has a golden dump and full-chunk gates on both EPs, so all three
models in the catalog are validated the same way and the second architecture is no longer
resting on a short-chunk spot check.

Export: 38.9 s, 644.4 MB, torch vs ORT-CPU max `1.013e-06` rms `8.277e-09`.

| gate | CPU EP | DirectML EP |
|---|---|---|
| `mask` | max `2.861e-06`, rms `1.157e-08`, p99.9 `1.192e-07` **OK** | max `1.937e-06`, rms `1.428e-08`, p99.9 `1.937e-07` **OK** |
| `synth` | 136.5 dB **OK** | 136.5 dB **OK** |
| `quality` vocals | ref -14.58, driver -14.93, delta -0.35 dB **OK** | same |
| `drift` | 28.6 dB (informational) | 28.6 dB |

| EP | ms / chunk (best) | median | chunk realtime | e2e, 12 s fixture |
|---|---|---|---|---|
| CPU | 24393.5 | 24438.4 | 0.33x | 122.60 s (0.10x) |
| DirectML | **2208.5** | 2209.9 | **3.62x** | **11.69 s (1.03x)** |

The golden dump is **not** committed, same call as the 4-stem one: the weights are not
redistributable, so anyone who can regenerate the dump already has the checkpoint, and
committing a dump derived from those weights would redistribute them by another route.

One thing the numbers say about the fixture rather than the model: the reference itself scores
**-14.58 dB** on the fixture's vocals. The gates prove the port matches the reference to
1.9e-06 on the mask, which is what they are for; they say nothing about how well this model
separates real music. See the fixture survey below.

### A realistic fixture
`refs/inputs/fixture_mix.wav` is synthetic and both the reference and the port score only
~1 dB SI-SDR on it, which makes the `quality` gate weak: it proves the port is not worse than
the reference, but it cannot detect a change that degrades separation on real music. Until
that is fixed, judge quality on real material by ear.

**Surveyed 2026-08-17; still blocked, and the reason is licensing, not effort.** What a fixture
has to be here is narrow: committable to a public repo, so redistributable under a licence that
permits it; real recorded music with real stems; and coherent, because unrelated sources mixed
together are *easier* to separate than a real arrangement, which would raise the score without
strengthening the gate. Nothing checked satisfies all three:

| Candidate | Licence | Why it fails |
|---|---|---|
| MUSDB18 / MUSDB18-HQ | CC BY-NC-SA, academic use only, access on request | not redistributable |
| MedleyDB | CC BY-NC-SA | not redistributable |
| MoisesDB | research licence | not redistributable |
| Cambridge-MT "Recording Secrets" backing stems | free for educational use | not redistributable |
| Open Multitrack Testbed (QMUL) | mixed CC | host is down — 403 over TLS, 502 over HTTP |
| Slakh2100 | **CC BY 4.0**, redistributable | rendered from MIDI, so no vocals at all; and it ships as a single 104 GB tarball with no per-track download |
| VocalSet | **CC BY 4.0**, redistributable | real singing, but a cappella technique exercises, and only 2–6 GB zips |

So the only two permissively licensed sources found are exactly the two halves that cannot be
made into one coherent song: VocalSet has voices and no music, Slakh has music and no voices.
Gluing them, or gluing public-domain instrumental recordings together, buys spectral realism
(real transients, real harmonic structure, real reverb tails) but not musical coherence — and
musical coherence is what the current gate is missing. That is why this has not been done: the
half-measure would make the numbers look better while leaving the weakness in place.

The one path that would actually work is commissioning or recording a short multitrack and
releasing it CC0, which is a project of its own.

### A publishable multi-stem checkpoint
`bs_roformer_musdb18_4stem` works (see Done) but cannot be redistributed, so nobody gets a
4-stem graph without a torch install and a 527 MB download. Worth noting alongside the
licence problem: in fp16 this model runs at 1.23x realtime instead of 0.13x, so the thing
blocking it is now purely the licence, not the cost of running it.

**Asked, 2026-08-17:**
[MSST#249](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/249) requests
permission to redistribute the ONNX graph of this one checkpoint — chosen because it is served
from ZFTurbo's own release rather than a third party's, which is exactly the distinction he drew
in #90 when he said he could not decide for models other people posted. Awaiting an answer; the
graph stays unpublished either way until there is one. The survey behind that call found
**no** 4-stem separator of roformer-tier quality with a redistributable weights licence. The
only genuinely permissive multi-stem model is Open-Unmix `umxhq` (MIT on its Zenodo record),
which is a BiLSTM at ~5.4 dB average SDR — a different architecture and a different quality
tier, so it is a separate port, not a variant of this one. Worth doing anyway if the goal is
"anyone can run 4 stems", because it is the only one that can actually ship.

### ~~Upstream the DirectML bug~~ — filed 2026-08-17
`Split(2) -> Sigmoid -> Mul` on the DirectML EP is reported as
[onnxruntime#32146](https://github.com/microsoft/onnxruntime/issues/32146), with the ~20-line
repro from `docs/onnxruntime-dml-glu-bug.md`. Anything exporting `nn.GLU` — a lot of audio and
NLP models — is silently affected, so the value is in someone else being able to reproduce it
without ever seeing this repo.

### Dynamic time axis
Not possible without replacing the rotary embedding: it caches its frequency table per sequence
length, so the traced graph is fixed at one chunk length. Exporting several fixed lengths is the
practical workaround (`--chunk`). A rewrite that computes the rotary table inside the graph would
lift this, and has not been attempted.
