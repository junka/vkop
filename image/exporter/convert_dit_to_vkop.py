"""Convert Qwen-Image-2.1 DiT ONNX graphs to VKOP format without optimization.

Large models (>13 GB external data) cannot pass through onnxoptimizer or
ConstantFolder because they try to serialize the full proto. This script
patches the optimizer to be a no-op and directly converts the DAG structure.

    /Users/doudou/qi21-env/bin/python convert_dit_to_vkop.py [prefill|decode]
"""

import sys
from pathlib import Path

# Add onnx2vkop package parent to path
ONNX2VKOP_PARENT = (Path(__file__).resolve().parent / ".." / ".." / "model" / "pypi").resolve()
sys.path.insert(0, str(ONNX2VKOP_PARENT))

import onnx
from onnx2vkop.converter import ModelConverter


def main(mode):
    path = Path(__file__).resolve().parent / f"dit_{mode}.onnx"
    print(f"[{mode}] loading {path.name}...")

    # Patch the optimizer to be a no-op for large models
    from onnx2vkop import optimizer as opt_module
    original_optimize = opt_module.ONNXOptimizer.optimize_model
    
    @staticmethod
    def patched_optimize(onnx_model, batch_size=1):
        print("[optimize] Skipping optimization (large model, >13GB weights)")
        return onnx_model  # Return unmodified model
    
    opt_module.ONNXOptimizer.optimize_model = patched_optimize
    
    # Now call parse_onnx_model which will use our patched optimizer
    converter = ModelConverter()
    dag_model = converter.parse_onnx_model(str(path), batch_size=1)
    
    if dag_model is None:
        print(f"[{mode}] ERROR: DAG build failed")
        sys.exit(1)
    
    print(f"[{mode}] DAG built: {len(dag_model.nodes)} nodes")
    
    # Collect operator statistics
    op_stats = {}
    for node in dag_model.nodes.values():
        op_type = node.op_type
        op_stats[op_type] = op_stats.get(op_type, 0) + 1
    
    print(f"\n[{mode}] Operator Statistics ({len(dag_model.nodes)} total nodes):")
    print(f"{'idx':<5} {'type':<25} {'count':<10}")
    for idx, (op_type, count) in enumerate(sorted(op_stats.items()), 1):
        print(f"{idx:<5} {op_type:<25} {count:<10}")
    
    # Try to save (will likely fail due to protobuf limits, but we get the stats)
    out_path = Path(__file__).resolve().parent / f"dit_{mode}.vkopbin"
    try:
        print(f"\n[{mode}] Attempting to save to {out_path.name}...")
        dag_model.save_to_binary(str(out_path))
        print(f"[{mode}] Saved successfully ({out_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"[{mode}] Binary save failed (expected for large models): {type(e).__name__}: {e}")
        print(f"[{mode}] But operator statistics collected successfully above")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "prefill"
    main(mode)
