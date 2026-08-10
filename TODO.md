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
- DirectML GLU miscomputation found, characterised with micro-graphs, and worked around.
- `mel_band_roformer_kim` (MIT weights) exported, validated, benchmarked and published in the
  `models-v1.0` release with `manifest.json` (source + graph SHA-256).

## Not done

### fp16 graph
Only fp32 is published. fp16 would halve the 931 MB download and the ~1.3 GB attention
intermediate, and DirectML handles fp16 well. Not attempted at all — no measurement exists of
what fp16 costs in mask accuracy for this architecture, and given how sensitive it is to
float32-level input differences (see README), that measurement is the whole job. Do not assume
it is free.

### BS-RoFormer golden + gates
`bs_roformer_viperx_1297` exports cleanly and its DirectML parity was verified at a short chunk
(T=101, max 6.7e-06), but no golden dump, no full-chunk validation and no bench exist for it —
its weights are not redistributable, so it was not the priority. Running
`capture_baseline.py` + `validate_ort.py` on it needs only the checkpoint and about 20 minutes
of CPU.

### A realistic fixture
`refs/inputs/fixture_mix.wav` is synthetic and both the reference and the port score only
~1 dB SI-SDR on it, which makes the `quality` gate weak: it proves the port is not worse than
the reference, but it cannot detect a change that degrades separation on real music. A better
fixture would be a short public-domain multitrack with real stems. Until then, judge quality on
real material by ear.

### Second stem / multi-stem checkpoints
The catalog only holds single-stem (`vocals`) models. The driver already returns
`[num_stems, C, N]` and `SpecOnlyBSRoformer` already stacks every mask estimator, so a 4-stem
checkpoint should work unchanged — but it has never been run. `RoformerSpec.stems` and the
golden capture assume one stem in a couple of places worth re-reading first.

### Upstream the DirectML bug
The `Split(2) -> Sigmoid -> Mul` miscomputation on the DirectML EP is reproducible in about 20
lines and is not reported anywhere upstream. It deserves an onnxruntime issue; anything that
exports `nn.GLU` (a lot of audio and NLP models) is silently affected.

### Dynamic time axis
Not possible without replacing the rotary embedding: it caches its frequency table per sequence
length, so the traced graph is fixed at one chunk length. Exporting several fixed lengths is the
practical workaround (`--chunk`). A rewrite that computes the rotary table inside the graph would
lift this, and has not been attempted.

### Upflow integration
Out of scope here by design. What it needs: a `RoformerSeparator` alongside
`mdx_separator.py` / `vr_deecho_separator.py`, reusing `OnnxStemSeparator` for the session cache,
cancellation and 44.1 kHz I/O. Two things do not map onto the MDX engine's assumptions —
these graphs emit one stem with the residual being a plain subtraction (no `compensate` factor),
and the model is ~10x heavier per second of audio than MDX (4.3x realtime vs 38x on the same
card), so it belongs behind an explicit "high quality, slow" choice rather than as a default.
