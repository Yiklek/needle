"""Integration tests for the DSL kernel system.

Tests the end-to-end flow:
1. Register a DSL kernel via Python
2. Create tensors via FineflowPyApi
3. Call the kernel through the runtime registry
4. Verify numerical correctness
"""

import sys
import numpy as np

sys.path.insert(0, "build")

import FineflowPyApi as lib


def test_register_and_call_cpu_kernel():
    """Test registering a DSL CPU kernel and calling it."""

    def double_compute(inputs, outputs):
        x = inputs[("in", 0)]
        y = outputs[("out", 0)]
        y[:] = x * 2.0

    lib.register_dsl_kernel("test_double", "cpu", double_compute)

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t0 = lib.from_numpy(a)

    result = lib.call_dsl_kernel("test_double", t0)

    expected = a * 2.0
    actual = lib.to_numpy(result)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_register_and_call_two_input_kernel():
    """Test registering a two-input DSL CPU kernel."""

    def add_compute(inputs, outputs):
        a = inputs[("in", 0)]
        b = inputs[("in", 1)]
        y = outputs[("out", 0)]
        np.add(a, b, out=y)

    lib.register_dsl_kernel("test_add", "cpu", add_compute)

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    t0 = lib.from_numpy(a)
    t1 = lib.from_numpy(b)

    result = lib.call_dsl_kernel2("test_add", t0, t1)

    expected = a + b
    actual = lib.to_numpy(result)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_gelu_kernel():
    """Test GELU activation via DSL kernel."""

    def gelu_compute(inputs, outputs):
        x = inputs[("in", 0)]
        y = outputs[("out", 0)]
        c1 = np.float32(0.044715)
        c2 = np.float32(0.7978845608028654)
        x3 = x * x * x
        inner = c2 * (x + c1 * x3)
        np.multiply(x * np.float32(0.5), np.float32(1.0) + np.tanh(inner), out=y)

    lib.register_dsl_kernel("test_gelu", "cpu", gelu_compute)

    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    t0 = lib.from_numpy(a)

    result = lib.call_dsl_kernel("test_gelu", t0)

    c1 = 0.044715
    c2 = np.sqrt(2.0 / np.pi)
    x = a
    x3 = x ** 3
    inner = c2 * (x + c1 * x3)
    expected = 0.5 * x * (1.0 + np.tanh(inner))

    actual = lib.to_numpy(result)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_register_and_call_cpu_kernel()
    print("PASS: test_register_and_call_cpu_kernel")

    test_register_and_call_two_input_kernel()
    print("PASS: test_register_and_call_two_input_kernel")

    test_gelu_kernel()
    print("PASS: test_gelu_kernel")

    print("\nAll DSL kernel tests passed!")
