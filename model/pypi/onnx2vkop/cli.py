#!/usr/bin/env python3
"""Command-line interface for onnx2vkop."""

import argparse
import sys
from pathlib import Path

try:
    from .converter import ModelConverter
except ImportError:
    from converter import ModelConverter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert and optimize ONNX models to VKOP format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  onnx2vkop -i model.onnx

  # With quantization
  onnx2vkop -i model.onnx -q fp16
  onnx2vkop -i model.onnx -q int8

  # With all optimizations
  onnx2vkop -i model.onnx -u -r -b 4
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input ONNX model file path")
    parser.add_argument(
        "-o", "--output", help="Output VKOP binary file path (default: input_name.vkopbin)"
    )
    parser.add_argument(
        "-q", "--quant", choices=["fp16", "int8"], help="Quantization type: fp16 or int8"
    )
    parser.add_argument(
        "-u", "--unify", action="store_true", help="Convert initializers to a single memory block"
    )
    parser.add_argument(
        "-r", "--rgba", action="store_true", help="NCHW to RGBA conversion for initializers"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=1, help="Batch size for inference (default: 1)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__import__('onnx2vkop').__version__}"
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    if not input_path.suffix.lower() in [".onnx"]:
        print("Error: Input file must be an ONNX model (.onnx)")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".vkopbin")

    print(f"Converting model: {input_path}")
    print(f"Output will be saved to: {output_path}")

    try:
        # Create converter and run conversion
        converter = ModelConverter()
        dag_model = converter.parse_onnx_model(str(input_path), args.batch)

        # Create a simple args object for apply_optimizations
        class Args:
            def __init__(self, quant, unify, rgba):
                self.quant = quant
                self.unify = unify
                self.rgba = rgba

        converter.apply_optimizations(dag_model, Args(args.quant, args.unify, args.rgba))

        # Save the converted model
        dag_model.save_to_binary(str(output_path))

        print("\nConversion completed successfully!")
        print(f"Output file size: {output_path.stat().st_size:,} bytes")

        # Print statistics
        op_stats = {}
        for node in dag_model.nodes.values():
            op_type = node.op_type
            op_stats[op_type] = op_stats.get(op_type, 0) + 1

        print("\nOperator Statistics:")
        print(f"{'idx':<5} {'type':<20} {'count':<10}")
        for idx, (op_type, count) in enumerate(op_stats.items(), 1):
            print(f"{idx:<5} {op_type:<20} {count:<10}")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
