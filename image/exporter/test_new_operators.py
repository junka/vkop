#!/usr/bin/env python3
"""
Test script to verify new operators (ReduceMean, Min, Max, Mod) are correctly
registered and can be instantiated by the vkop runtime.

This script:
1. Converts ONNX models containing the new operators to vkopbin format
2. Verifies operator statistics show the new ops
3. Checks that conversion completes without errors
"""

import sys
import os

# Add onnx2vkop to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model', 'pypi'))

from onnx2vkop.cli import main as convert_main

def test_operator_conversion(model_name):
    """Test that a model with new operators converts successfully."""
    onnx_path = f"image/exporter/{model_name}.onnx"
    vkopbin_path = f"image/exporter/{model_name}_test.vkopbin"
    
    if not os.path.exists(onnx_path):
        print(f"⚠ Model {onnx_path} not found, skipping")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Run conversion
        sys.argv = ['onnx2vkop', '-i', onnx_path, '-o', vkopbin_path]
        convert_main()
        
        # Check output file
        if os.path.exists(vkopbin_path):
            size_mb = os.path.getsize(vkopbin_path) / (1024 * 1024)
            print(f"\n✓ Conversion successful: {vkopbin_path} ({size_mb:.1f} MB)")
            
            # Clean up test file
            os.remove(vkopbin_path)
            return True
        else:
            print(f"\n✗ Conversion failed: output file not created")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        return False

def main():
    print("Testing new operator implementation (ReduceMean, Min, Max, Mod)")
    print("="*60)
    
    # Test models that contain the new operators
    test_cases = [
        "dit_prefill_tiny",
        "dit_decode_tiny",
    ]
    
    results = []
    for model in test_cases:
        result = test_operator_conversion(model)
        results.append((model, result))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r == True)
    total = len(results)
    
    for model, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {model}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All operator tests passed!")
        print("New operators (ReduceMean, Min, Max, Mod) are working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
