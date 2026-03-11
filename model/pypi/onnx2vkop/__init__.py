"""ONNX to VKOP Model Converter and Optimizer."""

__version__ = "0.1.0"
__author__ = "Wan Junjie"
__email__ = "wan.junjie@foxmail.com"

try:
    from .converter import ModelConverter
    from .dag import DAGBasedModel, Node
    from .optimizer import FusionOptimizer, InitializerMerger, ONNXOptimizer, Quantizer
except ImportError:
    from converter import ModelConverter
    from dag import DAGBasedModel, Node
    from optimizer import FusionOptimizer, InitializerMerger, ONNXOptimizer, Quantizer

__all__ = [
    "ModelConverter",
    "ONNXOptimizer",
    "FusionOptimizer",
    "InitializerMerger",
    "Quantizer",
    "Node",
    "DAGBasedModel",
    "__version__",
]
