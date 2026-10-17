"""Read-only inspector for a vkopbin file.

The C++ runtime's `dump_model()` already prints a summary when a model
loads, but that path drags up a Vulkan device and every pipeline the graph
uses — expensive for the "why did this pass fold X?" question that a
converted artifact should answer on its own. This module reads the
FlatBuffers structure directly through the generated Python bindings and
prints (or emits as JSON) the same shape of information without touching
Vulkan.

Typical uses:

  * "did my new fusion pass actually fire?" -- diff two `.vkopbin` files
    via `--json | diff` without a GPU
  * "what op types does this model use?" -- feeds the same shape the
    shader selection in shader_map.py operates on
  * "why does llm_chat take 4.9s to load?" -- the `--sizes` view groups
    initializers by bytes so a small number of huge blobs are obvious

The inspector never writes, and never requires the model to be loadable by
the runtime: a broken flatbuffer raises through the generated accessors
and the CLI turns it into a non-zero exit code with a clear message.

Backend / precision are properties the Driver picks at load time (vkop has
no per-model "this is a buffer model" flag -- the same .vkopbin can be
loaded either way by a caller), so the inspector reports what IS in the
file, not what a runtime would do with it.
"""
from __future__ import annotations

import argparse
import json
import os.path as _osp
import sys
from pathlib import Path
from typing import Dict, List, Optional

# The generated flatbuffers bindings reference `vkop.model.*` at top level
# (the namespace declared in vkop_model.fbs), so the folder holding the
# `vkop` package directory must be importable. Add it here instead of
# forcing every consumer to prepend to sys.path.
_GEN_ROOT = _osp.join(_osp.dirname(__file__), "generated")
if _GEN_ROOT not in sys.path:
    sys.path.insert(0, _GEN_ROOT)

try:
    from .generated.vkop.model import Model as _Model_mod
    from .generated.vkop.model import (
        Attribute,
        AttrType,
        ConcurrentLevel,
        InitializerEntry,
        Node,
        ShapeRef,
    )
except ImportError:
    from generated.vkop.model import Model as _Model_mod  # type: ignore
    from generated.vkop.model import (  # type: ignore
        Attribute,
        AttrType,
        ConcurrentLevel,
        InitializerEntry,
        Node,
        ShapeRef,
    )

_MODEL_VERSION = 3


def _read_bytes(fb, i, name_fn):
    """Fetch the i-th byte string from a flatbuffers accessor safely."""
    s = fb.ValueAt(i) if hasattr(fb, "ValueAt") else None
    return s


def _attr_type_name(code: int) -> str:
    return {
        AttrType.AttrType.String: "String",
        AttrType.AttrType.Int64: "Int64",
        AttrType.AttrType.Float32: "Float32",
        AttrType.AttrType.Ints: "Ints",
        AttrType.AttrType.Floats: "Floats",
        AttrType.AttrType.Tensor: "Tensor",
        AttrType.AttrType.Bool: "Bool",
    }.get(code, f"AttrType({code})")


class ModelView:
    """Structural snapshot of a vkopbin, produced once, reused for any view.

    The FlatBuffers accessors are zero-copy but each call re-walks the
    vtable; we snapshot everything once so `--json` and `--sizes` don't
    double the traversal cost on the 935-node llm.vkopbin.
    """

    def __init__(self, path: str):
        data = Path(path).read_bytes()
        if data[4:8] != b"VKOP":
            raise ValueError(
                f"{path}: not a VKOP model file (identifier bytes missing)"
            )
        self.root = _Model_mod.Model.GetRootAs(data, 0)
        if self.root.Version() != _MODEL_VERSION:
            raise ValueError(
                f"{path}: version {self.root.Version()} != supported "
                f"{_MODEL_VERSION}; reconvert with `python3 -m onnx2vkop.cli`"
            )
        self.path = path
        self.file_size = len(data)
        # External-blob mode: files >800MB append the initializer data AFTER
        # the flatbuffer itself, pointed at by an 8-byte little-endian
        # trailer. The bindings see an empty initializer_blob vector; the
        # trailer offset is the real blob start.
        self.external_blob_trailer: Optional[int] = None
        if self.root.InitializerBlobLength() == 0 and len(data) >= 8:
            # Best-effort: only treat the trailing 8 bytes as a trailer if
            # the model actually has initializers (i.e. it's not a
            # zero-weight graph).
            if self.root.InitializersLength() > 0:
                self.external_blob_trailer = int.from_bytes(
                    data[-8:], "little", signed=False
                )

    def shapes(self) -> List[Dict]:
        out = []
        for i in range(self.root.InputsLength()):
            sr = self.root.Inputs(i)
            if sr is None:
                continue
            out.append(self._shape_dict(sr))
        return out

    def outputs(self) -> List[Dict]:
        out = []
        for i in range(self.root.OutputsLength()):
            sr = self.root.Outputs(i)
            if sr is None:
                continue
            out.append(self._shape_dict(sr))
        return out

    def nodes(self) -> List[Dict]:
        out = []
        for i in range(self.root.NodesLength()):
            n = self.root.Nodes(i)
            if n is None:
                continue
            out.append(self._node_dict(n))
        return out

    def initializers(self) -> List[Dict]:
        out = []
        for i in range(self.root.InitializersLength()):
            ini = self.root.Initializers(i)
            if ini is None:
                continue
            dims = [ini.Dims(j) for j in range(ini.DimsLength())]
            out.append(
                {
                    "name": self._as_str(ini.Name()),
                    "dtype": self._as_str(ini.Dtype()),
                    "dims": dims,
                    "offset": ini.Offset(),
                    "size": ini.Size(),
                }
            )
        return out

    def concurrent_levels(self) -> List[List[str]]:
        out = []
        for i in range(self.root.ConcurrentLevelsLength()):
            lvl = self.root.ConcurrentLevels(i)
            if lvl is None:
                out.append([])
                continue
            names = []
            for j in range(lvl.NodesLength()):
                s = lvl.Nodes(j)
                names.append(self._as_str(s))
            out.append(names)
        return out

    def summary(self) -> Dict:
        nodes = self.nodes()
        op_counts: Dict[str, int] = {}
        for n in nodes:
            op_counts[n["op_type"]] = op_counts.get(n["op_type"], 0) + 1
        return {
            "path": self.path,
            "file_size_bytes": self.file_size,
            "version": self.root.Version(),
            "unified": bool(self.root.Unified()),
            "rgba": bool(self.root.Rgba()),
            "external_blob_trailer_offset": self.external_blob_trailer,
            "num_inputs": len(self.shapes()),
            "num_outputs": len(self.outputs()),
            "num_nodes": len(nodes),
            "num_initializers": len(self.initializers()),
            "num_concurrent_levels": len(self.concurrent_levels()),
            "op_type_counts": dict(
                sorted(op_counts.items(), key=lambda kv: -kv[1])
            ),
        }

    def sizes_breakdown(self) -> Dict:
        inits = self.initializers()
        total = sum(i["size"] for i in inits)
        by_dtype: Dict[str, Dict[str, int]] = {}
        for i in inits:
            d = by_dtype.setdefault(i["dtype"], {"count": 0, "bytes": 0})
            d["count"] += 1
            d["bytes"] += i["size"]
        biggest = sorted(inits, key=lambda i: -i["size"])[:10]
        return {
            "total_initializer_bytes": total,
            "by_dtype": by_dtype,
            "top_10_by_size": [
                {
                    "name": i["name"],
                    "dtype": i["dtype"],
                    "size": i["size"],
                    "dims": i["dims"],
                }
                for i in biggest
            ],
        }

    def _shape_dict(self, sr: "ShapeRef.ShapeRef") -> Dict:
        return {
            "name": self._as_str(sr.Name()),
            "dtype": self._as_str(sr.Dtype()),
            "dims": [sr.Dims(j) for j in range(sr.DimsLength())],
            "value_dynamic": bool(sr.ValueDynamic()),
        }

    def _node_dict(self, n: "Node.Node") -> Dict:
        ins = [
            self._shape_dict(n.Inputs(j)) for j in range(n.InputsLength())
        ]
        outs = [
            self._shape_dict(n.Outputs(j)) for j in range(n.OutputsLength())
        ]
        attrs = {}
        for k in range(n.AttributesLength()):
            a = n.Attributes(k)
            if a is None:
                continue
            attrs[self._as_str(a.Key())] = self._attr_value(a)
        deps = [self._as_str(n.Dependencies(j)) for j in range(n.DependenciesLength())]
        dets = [self._as_str(n.Dependents(j)) for j in range(n.DependentsLength())]
        return {
            "op_type": self._as_str(n.OpType()),
            "name": self._as_str(n.Name()),
            "attributes": attrs,
            "inputs": ins,
            "outputs": outs,
            "dependencies": deps,
            "dependents": dets,
        }

    def _attr_value(self, a: "Attribute.Attribute"):
        kind = a.Type()
        if kind == AttrType.AttrType.String:
            return self._as_str(a.Sval())
        if kind == AttrType.AttrType.Int64:
            return a.Ival()
        if kind == AttrType.AttrType.Float32:
            return a.Fval()
        if kind == AttrType.AttrType.Bool:
            return bool(a.Bval())
        if kind == AttrType.AttrType.Ints:
            return [a.Ints(i) for i in range(a.IntsLength())]
        if kind == AttrType.AttrType.Floats:
            return [a.Floats(i) for i in range(a.FloatsLength())]
        if kind == AttrType.AttrType.Tensor:
            td = a.Tval()
            if td is None:
                return None
            return {
                "dtype": self._as_str(td.Dtype()),
                "dims": [td.Dims(i) for i in range(td.DimsLength())],
                "bytes": td.DataLength(),
            }
        return None

    def _as_str(self, v) -> str:
        if v is None:
            return ""
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------


def print_text(view: ModelView, section: str = "all") -> None:
    s = view.summary()
    if section in ("all", "summary"):
        print(f"{s['path']}  ({s['file_size_bytes']:,} bytes)")
        print(f"  version={s['version']}  unified={s['unified']}  rgba={s['rgba']}")
        if s["external_blob_trailer_offset"]:
            print(
                f"  external data blob at offset "
                f"{s['external_blob_trailer_offset']:,}"
            )
        print(
            f"  {s['num_nodes']} nodes  {s['num_initializers']} initializers  "
            f"{s['num_concurrent_levels']} levels"
        )
        print("  op type census:")
        for op, cnt in s["op_type_counts"].items():
            print(f"    {op:<24s} {cnt}")
    if section in ("all", "inputs"):
        print("\nInputs:")
        for sh in view.shapes():
            dyn = "  dynamic-values" if sh["value_dynamic"] else ""
            print(
                f"  {sh['name']:<32s} {sh['dtype']:<10s} "
                f"{str(sh['dims']):<24s}{dyn}"
            )
        print("Outputs:")
        for sh in view.outputs():
            dyn = "  dynamic-values" if sh["value_dynamic"] else ""
            print(
                f"  {sh['name']:<32s} {sh['dtype']:<10s} "
                f"{str(sh['dims']):<24s}{dyn}"
            )
    if section in ("all", "nodes"):
        print("\nNodes:")
        for n in view.nodes():
            ins = ", ".join(i["name"] for i in n["inputs"])
            outs = ", ".join(o["name"] for o in n["outputs"])
            attr = f"  attrs={n['attributes']}" if n["attributes"] else ""
            print(f"  [{n['op_type']}] {n['name']}")
            print(f"      in:  {ins}")
            print(f"      out: {outs}{attr}")
    if section in ("all", "initializers"):
        print("\nInitializers:")
        for i in view.initializers():
            print(
                f"  {i['name']:<40s} {i['dtype']:<10s} "
                f"{str(i['dims']):<24s} off={i['offset']:>12,} "
                f"size={i['size']:>12,}"
            )
    if section in ("all", "levels"):
        print("\nConcurrent levels:")
        for k, names in enumerate(view.concurrent_levels()):
            joined = ", ".join(names)
            if len(joined) > 200:
                joined = joined[:197] + "..."
            print(f"  L{k} ({len(names):>3}): {joined}")
    if section == "sizes":
        print("\nSize breakdown:")
        sz = view.sizes_breakdown()
        print(f"  total initializer bytes: {sz['total_initializer_bytes']:,}")
        print("  by dtype:")
        for dtype, d in sorted(
            sz["by_dtype"].items(), key=lambda kv: -kv[1]["bytes"]
        ):
            print(
                f"    {dtype:<10s} {d['count']:>5} entries  "
                f"{d['bytes']:>15,} bytes"
            )
        print("  top-10 largest entries:")
        for e in sz["top_10_by_size"]:
            print(
                f"    {e['name'][:40]:<40s} {e['dtype']:<10s} "
                f"{e['size']:>15,} bytes  {e['dims']}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SECTIONS = ("all", "summary", "inputs", "nodes", "initializers", "levels", "sizes")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="onnx2vkop.inspect",
        description="Print the structure of a .vkopbin without loading a GPU",
    )
    p.add_argument("model", help="path to a .vkopbin file")
    p.add_argument(
        "--section",
        choices=_SECTIONS,
        default="all",
        help="restrict output to one section (default: all)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON dump instead of text",
    )
    p.add_argument(
        "--limit-nodes",
        type=int,
        default=None,
        help="in the nodes section, cap the printed entries (helps on the LLM's 935 nodes)",
    )
    args = p.parse_args(argv)

    try:
        view = ModelView(args.model)
    except FileNotFoundError:
        print(f"inspect: {args.model}: not found", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"inspect: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        payload = {
            "summary": view.summary(),
            "inputs": view.shapes(),
            "outputs": view.outputs(),
            "initializers": view.initializers(),
            "sizes": view.sizes_breakdown(),
            "levels": view.concurrent_levels(),
        }
        nodes = view.nodes()
        if args.limit_nodes is not None:
            nodes = nodes[: args.limit_nodes]
        payload["nodes"] = nodes
        json.dump(payload, sys.stdout, indent=2, default=str)
        print()
        return

    print_text(view, section=args.section)


if __name__ == "__main__":
    main()
