"""Built-in elementwise kernels defined with TileLang."""

import tilelang.language as TL
from needle.dsl.registry import register_tilelang_op


@register_tilelang_op("builtin_add", device_types=["metal"], dtypes=["float32"])
@TL.prim_func
def add_kernel(
    X: TL.Buffer((128,), "float32"),
    Y: TL.Buffer((128,), "float32"),
    Z: TL.Buffer((128,), "float32"),
):
    with TL.Kernel(128) as bx:
        Z[bx] = X[bx] + Y[bx]


@register_tilelang_op("builtin_scale", device_types=["metal"], dtypes=["float32"])
@TL.prim_func
def scale_kernel(
    X: TL.Buffer((128,), "float32"),
    S: TL.Buffer((1,), "float32"),
    Y: TL.Buffer((128,), "float32"),
):
    with TL.Kernel(128) as bx:
        Y[bx] = X[bx] * S[0]
