## benchmark

不会使用 CPU 做推理，CPU利用率低于2%，batch cmd 提交占用
measure 1000 rounds avgerage

download models from [onnx repo](https://github.com/onnx/models/tree/main/validated/vision/classification)

| GPU | Model | Operators |Precision | latency (ms) |
| --- | ----- | --------- | -------- | ----------- |
| A2000| mobilenet | 64 | fp32 | 3.32 |
| A2000| mobilenet | 64 | fp16 | 2.99 |
| A2000| resnet18 | 41 | fp32 | 8.06 |
| A2000| resnet18 | 41 | fp16 | 7.02 |
| A2000| resnet34 | 73 | fp32 | 14.81 |
| A2000| resnet34 | 73 | fp16 | 12.54 |
| A2000| resnet50 | 90 | fp32 | 18.79 |
| A2000| resnet50 | 90 | fp16 | 15.47 |
| A2000| resnet101 | 175 | fp32 | 33.50 |
| A2000| resnet101 | 175 | fp16 | 26.44 |
| A2000| resnet152 | 260 | fp32 | 46.99 |
| A2000| resnet152 | 260 | fp16 | 37.42 |
| T2000| resnet18 | 41 | fp16 | 17.04 |
| T2000| resnet34 | 73 | fp16 | 30.03 |
| T2000| resnet50 | 90 | fp16 | 35.30 |

## 模型转换
```
python3 model/onnx2vkop.py <model.onnx>
```

## run benchmark
```
benchmark/vkbench <model.vkopbin> <image.jpg>
```