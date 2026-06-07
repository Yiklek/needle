"""TileLang compilation orchestration + disk cache.

Calls TileLang lower() to compile PrimFunc -> target-specific source.
Caches to ~/.cache/needle/kernels/{hash}/
"""

import dataclasses
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tilelang.engine.lower import lower


@dataclasses.dataclass
class ParamMeta:
    name: str
    role: str       # "input" | "output"
    index: int
    dtype: str


@dataclasses.dataclass
class KernelArtifact:
    kernel_name: str
    target: str
    source_hash: str
    kernel_source: str
    entry_point: str
    params_meta: list[ParamMeta]
    cache_path: Optional[Path]


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "needle" / "kernels"


def _tir_hash(prim_func) -> str:
    return hashlib.sha256(prim_func.script().encode()).hexdigest()[:16]


def _extract_entry_point(kernel_source: str, kernel_name: str) -> str:
    match = re.search(r'kernel\s+void\s+(\w+)\s*\(', kernel_source)
    if match:
        return match.group(1)
    return f"{kernel_name}_kernel"


def _extract_params_meta(prim_func) -> list[ParamMeta]:
    params = []
    buffer_map = getattr(prim_func, 'buffer_map', {})
    for i, param in enumerate(prim_func.params):
        name = param.name
        buf = buffer_map.get(param, None)
        dtype_str = str(buf.dtype) if buf is not None else "float32"
        role = "input"
        params.append(ParamMeta(name=name, role=role, index=i, dtype=dtype_str))
    # 最后一个参数为输出
    if params:
        params[-1].role = "output"
        params[-1].index = 0
        for idx, p in enumerate(params[:-1]):
            p.index = idx
    return params


def _load_cache(source_hash: str, target: str, cache_dir: Path) -> Optional[KernelArtifact]:
    cache_path = cache_dir / source_hash
    ext = "metal" if target == "metal" else target
    manifest_path = cache_path / "manifest.json"
    source_path = cache_path / f"kernel.{ext}"
    if not manifest_path.exists() or not source_path.exists():
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("target") != target:
        return None
    return KernelArtifact(
        kernel_name=manifest["kernel_name"],
        target=manifest["target"],
        source_hash=source_hash,
        kernel_source=source_path.read_text(),
        entry_point=manifest["entry_point"],
        params_meta=[ParamMeta(**p) for p in manifest["params"]],
        cache_path=source_path,
    )


def _save_cache(artifact: KernelArtifact, cache_dir: Path) -> Path:
    cache_path = cache_dir / artifact.source_hash
    cache_path.mkdir(parents=True, exist_ok=True)
    ext = "metal" if artifact.target == "metal" else artifact.target
    source_path = cache_path / f"kernel.{ext}"
    source_path.write_text(artifact.kernel_source)
    manifest = {
        "kernel_name": artifact.kernel_name,
        "target": artifact.target,
        "entry_point": artifact.entry_point,
        "source_hash": artifact.source_hash,
        "tir_hash": artifact.source_hash,
        "params": [dataclasses.asdict(p) for p in artifact.params_meta],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return source_path


def compile_kernel(prim_func, *, name: str, target: str,
                   cache_dir: Optional[str] = None) -> KernelArtifact:
    cache_path = Path(cache_dir) if cache_dir else _default_cache_dir()
    source_hash = _tir_hash(prim_func)

    cached = _load_cache(source_hash, target, cache_path)
    if cached is not None:
        return cached

    artifact = lower(prim_func, target=target)
    kernel_source = artifact.kernel_source
    entry_point = _extract_entry_point(kernel_source, name)
    params_meta = _extract_params_meta(prim_func)

    result = KernelArtifact(
        kernel_name=name, target=target, source_hash=source_hash,
        kernel_source=kernel_source, entry_point=entry_point,
        params_meta=params_meta, cache_path=None,
    )
    result.cache_path = _save_cache(result, cache_path)
    return result
