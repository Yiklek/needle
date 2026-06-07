# needle.dsl — DSL Kernel Integration Layer
#
# Provides the @register_tilelang_op decorator and companion APIs
# for defining operators in TileLang and integrating them with
# the Fineflow C++ runtime.

from needle.dsl.registry import register_tilelang_op
from needle.dsl.compiler import compile_kernel, KernelArtifact, ParamMeta
from needle.dsl.codegen import write_artifact, read_artifact, artifact_to_metal_source

__all__ = [
    "register_tilelang_op",
    "compile_kernel",
    "KernelArtifact",
    "ParamMeta",
    "write_artifact",
    "read_artifact",
    "artifact_to_metal_source",
]
