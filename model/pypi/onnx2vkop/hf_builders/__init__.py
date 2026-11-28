"""Per-architecture graph builders for `onnx2vkop --hf-checkpoint`.

Each module exposes

  * `make_wrapper(model)` -- the traceable module, for a real OR a meta HF
    model; the llm/exporter/export/*_export_onnx.py scripts export from it, so
    a script and the CLI cannot drift apart;
  * `build(config, torch_dtype, index) -> hf_import.GraphSpec` -- the meta
    trace entry the CLI calls.

`REGISTRY` maps config.json `model_type` to a module; `--hf-builder NAME`
forces one by its module name.
"""
from __future__ import annotations

import importlib
from typing import Callable

# model_type -> builder module (relative to this package)
REGISTRY = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "llama",
    "qwen3": "llama",
    "phi3": "phi3",
    "glm": "glm",
    "qwen3_vl": "qwen3vl",
    "qwen3_vl_text": "qwen3vl",
    "qwen2_moe": "qwen2_moe",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
}

DEFAULT = "llama"


def _module(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def by_name(name: str) -> Callable:
    names = sorted(set(REGISTRY.values()))
    if name not in names:
        from ..hf_import import HFImportError
        raise HFImportError(f"--hf-builder {name!r}: not a registered builder "
                            f"({', '.join(names)}) nor pkg.module:fn")
    return _module(name).build


def for_config(config) -> Callable:
    """The builder for config.model_type (outer type first, then the text
    config's), else the Llama one, which refuses a non-Llama block."""
    types = [getattr(config, "model_type", None)]
    if hasattr(config, "get_text_config"):
        types.append(getattr(config.get_text_config(), "model_type", None))
    for t in types:
        if t in REGISTRY:
            return _module(REGISTRY[t]).build
    return _module(DEFAULT).build
