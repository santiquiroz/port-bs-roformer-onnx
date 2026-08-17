# fp16 on DirectML: 12x on one architecture, 1.25 dB on the other

Measured 2026-08-17 on Windows 11, Radeon RX 7800 XT, `onnxruntime-directml==1.24.4`,
against the golden dumps in `refs/golden/`. The tooling is `toolkit/export_fp16.py`;
every number below is reproducible with the commands in it.

The TODO said fp16 was "not attempted at all" and warned that measuring the cost in mask
accuracy *was* the job. It is measured now, and the answer is not one answer: fp16 is
transformative for `bs_roformer_musdb18_4stem` and not usable for `mel_band_roformer_kim`.

## Getting a graph that loads at all

Three mechanical problems come before any numerics, and each one produces an error that
does not name its cause:

| Symptom | Cause | Fix |
|---|---|---|
| `Type parameter (T) of Optype (Mul) bound to different types` at the first attention block | `onnxconverter_common` leaves mixed dtypes on nodes fed by one converted and one non-converted producer | use `onnxruntime.transformers.float16` instead |
| `Type (tensor(float16)) of output arg (.../Cast_1_output_0) does not match expected type (tensor(float))` | the converter rewrites tensor types but leaves the old `value_info` | drop `graph.value_info` and let ORT re-infer |
| `Nodes in a graph must be topologically sorted` | the converter appends its `graph_input_cast_*` nodes after the nodes that consume them | `OnnxModel.topological_sort()` |

And one trap that produces no error at all: `onnx.save` **appends** to an existing
external-data file instead of truncating it. A second conversion run silently wrote a
506 MiB graph holding the same weights as the 255 MiB one, and it loaded fine.

## RMSNorm has to stay in fp32 or the mask is NaN

A pure fp16 graph loads, creates its DirectML session in 2.8 s and runs at full speed —
and returns an all-NaN mask, so every quality gate reads `-inf`.

The cause is RMSNorm, not attention: `Pow` squares a spectrum that peaks around 161 and
`ReduceMean` sums those squares across the block, which passes fp16's 65504 ceiling long
before the divide brings it back. Keeping `Pow`, `ReduceMean`, `Sqrt` and `Div` in fp32
costs 0.1 MiB of the 255 MiB and removes the NaN entirely.

Blocking the attention (`Einsum`, `Softmax`) *on top of that* changes nothing measurable:
-0.66 dB vs -0.64 dB on `other`. It is not in the default block list, and the intuition
that attention is where fp16 breaks is wrong for this architecture.

`mel_band_roformer_kim` needs `ScatterElements` blocked as well, for a different reason —
not precision but a missing kernel: `MLFloat16 data type is not supported with
ScatterElements opset 16 when reduction is 'add'`, and that node (the mel overlap-averaging
scatter, baked into the graph) is one ORT assigns to the CPU EP.

## `bs_roformer_musdb18_4stem`: 12x faster, half the size, 0.05 dB

| | fp32 | fp16 |
|---|---:|---:|
| size | 508.0 MiB | **254.9 MiB** |
| ms / chunk (best) | 17235.3 | **1431.7** |
| end to end, 12 s fixture | 94.19 s (0.13x) | **9.76 s (1.23x)** |

| quality | reference | fp16 driver | delta |
|---|---:|---:|---:|
| drums | 18.92 dB | 18.89 dB | -0.03 dB |
| bass | -7.29 dB | -7.36 dB | -0.06 dB |
| vocals | 17.65 dB | 17.59 dB | -0.06 dB |
| other | -7.72 dB | -8.36 dB | **-0.64 dB** |

12x is far more than the ~2x that halving the arithmetic buys, and that gap is the finding:
this graph holds a ~1.3 GB attention intermediate at T=1101 with four stems, so in fp32 it
spends most of its time moving memory it cannot keep resident. Halving the intermediate is
what turns "eight times slower than realtime" into "faster than realtime".

The single failing stem is worth reading carefully rather than counting. `other` is the one
the reference *itself* scores below zero on — it is not separated in the first place — and
the gate margin is an absolute 0.5 dB applied to stems whose reference scores span 18.92 dB
to -7.72 dB. The three stems the model actually separates lose 0.03-0.06 dB.

## `mel_band_roformer_kim`: 2.6x faster, half the size, and a real accuracy loss

| | fp32 | fp16 |
|---|---:|---:|
| size | 887.9 MiB | **444.5 MiB** |
| ms / chunk (best) | 1852.2 | **725.1** |
| end to end, 12 s fixture | 10.14 s (1.18x) | **4.23 s (2.83x)** |

| quality | reference | fp16 driver | delta |
|---|---:|---:|---:|
| vocals | 0.71 dB | -0.54 dB | **-1.25 dB** |

This one does not pass, and the drift number says it independently of the fixture's weak
ground truth: the fp16 output agrees with the **fp32 output** at only 14.4 dB SI-SDR, where
the 4-stem model's well-separated stems agree at 30-40 dB. That is a different result, not a
noisy one.

**Nothing fp16 is published.** The model that gains the most cannot be redistributed at all
(see the licence finding in the README), and the model that can be redistributed loses more
accuracy than the gate allows. Publishing the fast one would mean shipping a quality
regression to everyone who takes the default.

### The loss is spread, not one node

Three targeted attempts to buy `kim`'s accuracy back, each keeping more of the graph in
fp32, all landed in the same place:

| kept in fp32 on top of RMSNorm + ScatterElements | size | vocals delta | drift |
|---|---:|---:|---:|
| nothing | 444.5 MiB | -1.25 dB | 14.4 dB |
| `MatMul` | 879.4 MiB | -1.31 dB | 14.0 dB |
| `Sigmoid`, `Split` (the GLU path) | 444.6 MiB | -1.31 dB | 14.2 dB |

The `MatMul` row is the informative one. It leaves almost every weight in the model at fp32
— the graph barely shrinks, 1.0% — and the error does not move. So this is not a few
weights rounding badly and not one hot node: it is fp16 rounding accumulating through a deep
stack, and the mel filterbank, which sums overlapping bands, is a plausible amplifier that
`bs_roformer`'s plain band split does not have. There is no block list that fixes this while
keeping the point of the exercise.

## What was tried and did not work

- **Attention in fp32** (`Einsum`, `Softmax`): no measurable change, on either model.
- **Automatic mixed-precision search** (`onnxconverter_common.auto_mixed_precision_model_path`,
  which bisects for the smallest set of nodes that must stay fp32): aborts with
  `Validation failed for model with nothing converted to fp16`, i.e. its own baseline check
  fails before any conversion happens. That is not a tolerance problem on this hardware —
  measured separately, this graph on the DirectML EP is bit-deterministic: two runs in one
  session and a run in a fresh session all agree to **exactly 0.0**. The search is worth
  retrying against a fixed or patched converter; `toolkit/auto_fp16.py` keeps the setup.
