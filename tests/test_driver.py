"""Driver contract: torch-free, and the chunk schedule matches the reference."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

from driver.chunking import iter_chunks, plan_chunks, unpad_result, windowing_array
from driver.pipeline import RoformerDriver, RoformerSpec, complex_multiply

DRIVER = Path(__file__).resolve().parents[1] / "driver"


def test_driver_imports_no_torch():
    banned = {"torch", "librosa", "onnxruntime", "einops"}
    for path in DRIVER.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (roots & banned), f"{path.name} imports {roots & banned}"


def test_driver_runs_without_torch_in_a_clean_interpreter():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import numpy as np\n"
        "from driver.pipeline import RoformerDriver, RoformerSpec\n"
        "spec = RoformerSpec(chunk_size=4410, n_fft=256, hop_length=64, stems=('vocals',))\n"
        "d = RoformerDriver(lambda s: np.ones((1, 1, *s.shape[1:]), dtype=np.float32) * "
        "np.array([1.0, 0.0], dtype=np.float32), spec)\n"
        "out = d.separate(np.zeros((2, 9000), dtype=np.float32))\n"
        "print(out.shape)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(DRIVER.parent), check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "(1, 2, 9000)" in result.stdout


def test_windowing_array_matches_reference_example():
    """MSST's own docstring example: window_size=10, fade_size=3."""
    window = windowing_array(10, 3)
    expected = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
    assert np.allclose(window, expected)


def test_chunk_plan_covers_signal_with_unit_weight():
    plan = plan_chunks(300_000, chunk_size=100_000, num_overlap=2)
    mix = np.zeros((2, plan.padded_length))
    counter = np.zeros((1, 2, plan.padded_length))
    result = np.zeros_like(counter)
    for start, seg_len, _, window in iter_chunks(mix, plan):
        counter[..., start : start + seg_len] += window[:seg_len]
    out = unpad_result(result, counter, plan)
    assert out.shape[-1] == 300_000
    assert np.all(counter[..., : plan.padded_length] > 0)


def test_complex_multiply_matches_numpy_complex():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5, 2))
    b = rng.standard_normal((4, 5, 2))
    got = complex_multiply(a, b)
    want = (a[..., 0] + 1j * a[..., 1]) * (b[..., 0] + 1j * b[..., 1])
    assert np.abs(got[..., 0] - want.real).max() < 1e-12
    assert np.abs(got[..., 1] - want.imag).max() < 1e-12


def test_identity_mask_reconstructs_the_mix():
    """A mask of 1+0j must give back the input (minus the DC bin)."""
    spec = RoformerSpec(chunk_size=8820, n_fft=512, hop_length=128, stems=("vocals",),
                        zero_dc=False)
    rng = np.random.default_rng(1)
    mix = rng.standard_normal((2, 20000)).astype(np.float32) * 0.1

    def identity(s):
        mask = np.zeros((1, 1, *s.shape[1:]), dtype=np.float32)
        mask[..., 0] = 1.0
        return mask

    out = RoformerDriver(identity, spec).separate(mix)
    assert out.shape == (1, 2, 20000)
    assert np.abs(out[0] - mix).max() < 1e-4
