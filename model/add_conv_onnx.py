import torch
import torch.nn as nn
import torch.nn.functional as F

class AddConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(AddConvModel, self).__init__()
        # 可学习的加法偏置（形状与输入通道一致，用于逐通道加）
        self.add_bias = nn.Parameter(torch.randn(in_channels, 1, 1))  # [C, 1, 1] 广播到 [B, C, H, W]
        # 卷积层
        # Calculate groups to ensure divisibility: use 1 for regular conv, or in_channels for depthwise
        if in_channels == out_channels:
            groups = in_channels  # Depthwise convolution
        else:
            groups = 1  # Regular convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=True
        )

    def forward(self, x1, x2):
        # Add: x + bias (broadcasted)
        x = x1 + x2
        # Conv
        x = self.conv(x)
        return x

def main():
    # 创建模型实例
    model = AddConvModel(in_channels=16, out_channels=16, kernel_size=3)
    model.eval()  # 设置为 eval 模式（避免 BatchNorm/Dropout）

    dummy_input1 = torch.randn(1, 16, 8, 8)
    dummy_input2 = torch.randn(1, 16, 8, 8)

    onnx_path = "add_conv_model.onnx"
    torch.onnx.export(
        model,
        (dummy_input1, dummy_input2),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_x1', 'input_x2'],
        output_names=['output'],
    )
    print(f" ONNX 模型已保存至: {onnx_path}")

if __name__ == "__main__":
    main()