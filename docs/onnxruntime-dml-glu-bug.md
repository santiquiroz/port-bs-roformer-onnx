# DirectML EP miscomputes the `Split -> Sigmoid -> Mul` pattern emitted by `nn.GLU`

## Summary

The DirectML execution provider silently miscomputes the two-way `Split -> Sigmoid -> Mul`
pattern emitted when `torch.nn.GLU` is exported to ONNX. The CPU execution provider matches
PyTorch and NumPy. On DirectML, one split output is reused where the other is required; in the
characterized RoFormer graph this behaved as `a * sigmoid(a)` instead of `a * sigmoid(b)`. No
error or fallback is reported.

The diagnostic fingerprint is a minimum output value of approximately `-0.2785`, the minimum of
`x * sigmoid(x)`. Disabling ONNX Runtime graph optimizations does not change the result.

## Environment

- Windows 11 Pro 25H2, build 26200.9168
- AMD Radeon RX 7800 XT
- Python 3.11
- `onnxruntime-directml==1.24.4`
- `torch==2.13.0+cpu`
- `onnx==1.22.0`
- `numpy==2.4.6`

## Minimal reproduction

```python
import io
import numpy as np
import onnxruntime as ort
import torch

n = 4096
a = np.zeros((1, n), dtype=np.float32)
b = np.linspace(-4, 4, n, dtype=np.float32)[None]
x = np.concatenate((a, b), axis=-1)
model = io.BytesIO()
torch.onnx.export(torch.nn.GLU(-1), torch.from_numpy(x), model,
                  opset_version=17, dynamo=False)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
expected = a * (1 / (1 + np.exp(-b)))
for ep in ("CPUExecutionProvider", "DmlExecutionProvider"):
    session = ort.InferenceSession(model.getvalue(), opts, providers=[ep])
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]
    print(ep, "max_err=", np.max(np.abs(actual - expected)), "min=", actual.min())
```

Output:

```text
CPUExecutionProvider max_err= 0.0 min= 0.0
DmlExecutionProvider max_err= 3.928055 min= -0.27846456
```

## Expected behavior

`DmlExecutionProvider` should match the CPU execution provider and compute the GLU output from
the first half multiplied by the sigmoid of the second half. In the reproduction above, the
first half (`a`) is zero, so every output value should be zero.

## Actual behavior

DirectML returns non-zero values with a minimum of `-0.27846456`. This is the characteristic
minimum of a self-gated tensor, `x * sigmoid(x)`. With the input ordering in this minimal export,
the output behaves as `b * sigmoid(b)` instead of `a * sigmoid(b)`, demonstrating that the split
output feeding `Sigmoid` is also reused by `Mul`. In the original RoFormer graph, the equivalent
aliasing manifested as `a * sigmoid(a)` instead of `a * sigmoid(b)`.

## Characterization

The behavior was isolated with small ONNX graphs:

| Pattern | CPU EP | DirectML EP |
| --- | --- | --- |
| Two-way `Split`, `Sigmoid` on one output, then `Mul` with the other | Correct | Incorrect |
| Same graph with a positive axis instead of `-1` | Correct | Incorrect |
| Split operands exchanged | Correct | Correct |
| `Add` instead of `Mul` | Correct | Correct |
| Four-way split with two GLUs summed | Correct | Correct |
| 62-way split and concat used by the band-split block | Correct | Correct |
| Two `Slice` operations instead of the two-way `Split` | Correct | Correct |

The incorrect result is unchanged with
`SessionOptions.graph_optimization_level = ORT_DISABLE_ALL`, so the issue does not appear to be
caused by an ONNX Runtime graph rewrite or fusion.

## Workaround and model-level impact

The port replaces each `nn.GLU` during export with two `Slice` operations followed by `Sigmoid`
and `Mul` (`toolkit/spec_models.py`). The release `manifest.json` records `glu_replaced=60`.
With that workaround, DirectML parity against the unmodified PyTorch model improved from a maximum
absolute error of `5.76` to `6.7e-06`.

This can silently affect other ONNX models exported from `nn.GLU`, an operator pattern used in
audio and NLP architectures. The observed scope is the environment above; other DirectML hardware
and driver combinations have not yet been characterized.
