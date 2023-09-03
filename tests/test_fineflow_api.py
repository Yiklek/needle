import sys
import itertools
import numpy as np
import pytest
import FineflowPyApi as lib
import numpy as np

SHAPES = [
    (1,),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 2, 2),
    (2, 3, 2),
    (2, 3, 3),
    (2, 3, 3, 4),
    (100, 100, 100, 100),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_numpy_convert(shape):
    np.random.seed(0)
    a = np.random.randn(*shape)
    t = lib.from_numpy(a)
    n = lib.to_numpy(t)
    np.testing.assert_allclose(n, a, atol=1e-5, rtol=1e-5)


SCALAR_OPS = {
    "ewise_add": lib.ewise_add,
}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]


@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", SHAPES)
def test_op(fn, shape):
    np.random.seed(0)
    a0 = np.random.randn(*shape).astype("float32")
    a1 = np.random.randn(*shape).astype("float32")
    t0 = lib.from_numpy(a0)
    t1 = lib.from_numpy(a1)
    r = fn(t0, t1)
    np.testing.assert_allclose(lib.to_numpy(r), a0 + a1, atol=1e-5, rtol=1e-5)
