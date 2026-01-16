import os
import torch
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
    Inception_V3_Weights,
    MobileNet_V2_Weights,
    DenseNet121_Weights, DenseNet161_Weights, DenseNet201_Weights,
    EfficientNet_B0_Weights, EfficientNet_B7_Weights,
    ShuffleNet_V2_X1_0_Weights,
    SqueezeNet1_0_Weights,
    VGG16_Weights,
    AlexNet_Weights
)

OUTPUT_DIR = "onnx_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SPECS = [
    ("resnet18", models.resnet18, ResNet18_Weights.DEFAULT, 224),
    ("resnet34", models.resnet34, ResNet34_Weights.DEFAULT, 224),
    ("resnet50", models.resnet50, ResNet50_Weights.DEFAULT, 224),
    ("resnet101", models.resnet101, ResNet101_Weights.DEFAULT, 224),
    ("resnet152", models.resnet152, ResNet152_Weights.DEFAULT, 224),
    ("inception_v3", models.inception_v3, Inception_V3_Weights.DEFAULT, 299),
    ("mobilenet_v2", models.mobilenet_v2, MobileNet_V2_Weights.DEFAULT, 224),
    ("densenet121", models.densenet121, DenseNet121_Weights.DEFAULT, 224),
    ("densenet161", models.densenet161, DenseNet161_Weights.DEFAULT, 224),
    ("densenet201", models.densenet201, DenseNet201_Weights.DEFAULT, 224),
    ("efficientnet_b0", models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 224),
    ("efficientnet_b7", models.efficientnet_b7, EfficientNet_B7_Weights.DEFAULT, 600),
    ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.DEFAULT, 224),
    ("squeezenet1_0", models.squeezenet1_0, SqueezeNet1_0_Weights.DEFAULT, 224),
    ("vgg16", models.vgg16, VGG16_Weights.DEFAULT, 224),
    ("alexnet", models.alexnet, AlexNet_Weights.DEFAULT, 224),
]

def export_model(name, model_fn, weights, input_size):
    print(f"Loading {name}...")
    if name == "inception_v3":
        model = model_fn(weights=weights, transform_input=False)
        model.aux_logits = False
    else:
        model = model_fn(weights=weights)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)
    onnx_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    
    batch = torch.export.Dim("batch", min=1, max=64)  # Optional range limits

    print(f"Exporting {name} (opset=18) → {onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            external_data=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {0: batch}},
        )
        print(f"{name} exported successfully!")
    except Exception as e:
        print(f"{name} failed: {e}")

def main():
    print(f"Start exporting classic models to ONNX (save to ./{OUTPUT_DIR}/)")
    for name, fn, w, size in MODEL_SPECS:
        export_model(name, fn, w, size)
    print(f"\nCompleted! Total {len(MODEL_SPECS)} models.")

if __name__ == "__main__":
    main()