"""Artifact formatting — wraps TileLang compilation output for Fineflow.

Pure wrapper. TileLang does the compilation. This module formats the output.
"""

import json
from pathlib import Path

from needle.dsl.compiler import KernelArtifact, ParamMeta


def write_artifact(artifact: KernelArtifact) -> Path:
    """Returns the cache path (compiler.compile_kernel already wrote it)."""
    if artifact.cache_path is None:
        raise ValueError("Artifact has no cache_path")
    return artifact.cache_path


def read_artifact(cache_path: Path) -> KernelArtifact:
    """Read an artifact from a cache path."""
    manifest_path = cache_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return KernelArtifact(
        kernel_name=manifest["kernel_name"],
        target=manifest["target"],
        source_hash=manifest["source_hash"],
        kernel_source=cache_path.read_text(),
        entry_point=manifest["entry_point"],
        params_meta=[ParamMeta(**p) for p in manifest["params"]],
        cache_path=cache_path,
    )


def artifact_to_metal_source(artifact: KernelArtifact) -> str:
    """Return the .metal source code string."""
    if artifact.target != "metal":
        raise ValueError(f"Artifact target is '{artifact.target}', not 'metal'")
    return artifact.kernel_source
