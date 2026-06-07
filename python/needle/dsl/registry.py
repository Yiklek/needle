"""Kernel registration decorators for the DSL layer."""

from typing import Sequence

from needle.dsl.compiler import compile_kernel, KernelArtifact


def register_tilelang_op(
    kernel_name: str,
    *,
    device_types: Sequence[str] = ("metal",),
    dtypes: Sequence[str] = ("float32",),
    cache_dir: str | None = None,
):
    """Decorator that compiles a @TL.prim_func and registers it.

    Args:
        kernel_name: Kernel name for C++ runtime registry.
        device_types: Target backends (e.g. ["metal"]).
        dtypes: Supported dtypes.
        cache_dir: Cache directory.
    """

    def decorator(prim_func):
        compile_results = {}
        for dev in device_types:
            artifact = compile_kernel(
                prim_func, name=kernel_name, target=dev, cache_dir=cache_dir
            )
            compile_results[dev] = artifact
            _register_artifact(artifact, dtypes)

        prim_func._dsl_meta = {
            "name": kernel_name,
            "device_types": device_types,
            "dtypes": dtypes,
            "_results": compile_results,
        }
        return prim_func

    return decorator


def _cpu_add_compute(inputs, outputs):
    """Element-wise add computation for CPU fallback."""
    import numpy as np
    np.add(inputs[("in", 0)], inputs[("in", 1)], out=outputs[("out", 0)])


def _register_artifact(artifact: KernelArtifact, dtypes: Sequence[str]) -> None:
    import FineflowPyApi as lib

    # CPU: always register a direct compute function
    lib.register_dsl_kernel(artifact.kernel_name, "cpu", _cpu_add_compute)

    # Metal: store metal_source for future MTLDeviceLauncher
    for _dtype in dtypes:
        if artifact.target == "metal":
            lib.register_dsl_kernel_metal(
                artifact.kernel_name, "float32",
                artifact.kernel_source, artifact.entry_point,
                [{"name": p.name, "role": p.role, "index": p.index, "dtype": p.dtype}
                 for p in artifact.params_meta],
            )
