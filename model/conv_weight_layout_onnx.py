"""Source of truth for tests/ConvWeightLayoutTest.cpp's fixture.

Two Convs with constant weights of different rank:

  x4 [1,3,8,8]    x Conv(W_r4 [16,3,3,3],   kernel 3x3,   strides 1)   -> y4
  x5 [1,3,8,8,8]  x Conv(W_r5 [16,3,2,3,3], kernel 2x3x3, strides 1)   -> y5

The rank-4 case is the ordinary one: the image conv shader addresses its
weight as [OC, IC, KH, KW] through the transposed batch/chan mapping, so the
converter writes weight_transpose=1 / weight_pack=1. The rank-5 case is the
bug this fixture pins: the runtime used to apply set_transpose() to every
Conv-consumed weight and gate set_pack() on rank==4, so a Conv3d weight got
transpose=1 and Tensor::getGPUShape invented a batch/chan for it. Neither flag
may be set on a rank-5 weight now.

Kernel != stride on both Convs, so fold_single_patch_conv -- which requires
strides == kernel -- leaves them for ConvWeightLayoutPass to annotate.

Run with the venv interpreter (the runtime half needs no torch, only onnx):
    /home/gwm/.venv/bin/python model/conv_weight_layout_onnx.py
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

ONNX_PATH = "model/conv_weight_layout.onnx"


def main():
    rng = np.random.default_rng(1024)
    w4 = rng.standard_normal((16, 3, 3, 3), dtype=np.float32) * 0.1
    w5 = rng.standard_normal((16, 3, 2, 3, 3), dtype=np.float32) * 0.1

    inputs = [
        helper.make_tensor_value_info("x4", TensorProto.FLOAT, [1, 3, 8, 8]),
        helper.make_tensor_value_info("x5", TensorProto.FLOAT, [1, 3, 8, 8, 8]),
    ]
    outputs = [
        helper.make_tensor_value_info("y4", TensorProto.FLOAT, [1, 16, 6, 6]),
        helper.make_tensor_value_info("y5", TensorProto.FLOAT, [1, 16, 7, 6, 6]),
    ]
    initializers = [
        helper.make_tensor("W_r4", TensorProto.FLOAT, w4.shape,
                           w4.reshape(-1).tolist()),
        helper.make_tensor("W_r5", TensorProto.FLOAT, w5.shape,
                           w5.reshape(-1).tolist()),
    ]

    def conv(name, x, weight, out, kernel, strides):
        return helper.make_node(
            "Conv", [x, weight], [out], name=name,
            kernel_shape=list(kernel), strides=list(strides),
            pads=[0] * (2 * len(kernel)), dilations=[1] * len(kernel),
            group=1,
        )

    graph = helper.make_graph(
        [conv("conv_r4", "x4", "W_r4", "y4", (3, 3), (1, 1)),
         conv("conv_r5", "x5", "W_r5", "y5", (2, 3, 3), (1, 1, 1))],
        "conv_weight_layout", inputs, outputs, initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, ONNX_PATH)
    print(f"saved {ONNX_PATH}: conv_r4 W_r4 {w4.shape}, conv_r5 W_r5 {w5.shape}")


if __name__ == "__main__":
    main()
