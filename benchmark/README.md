## benchmark

不会使用 CPU 做推理，CPU利用率低于2%，batch cmd 提交占用
measure 1000 rounds avgerage

download models from using script:
```
python3 model/download_models.py
```

convert models to vkopbin
```
python3 model/onnx2vkop.py -i onnx_models/xxxx.onnx
```


| GPU | Model | Operators |Precision | latency (ms) |
| --- | ----- | --------- | -------- | ----------- |
| A2000| densenet121 | 246 | fp32 | 25.45 |
| A2000| densenet121 | 246 | fp16 | 22.26 |
| A2000| densenet161 | 326 | fp32 | 55.83 |
| A2000| densenet161 | 326 | fp16 | 46.75 |
| A2000| densenet201 | 406 | fp32 | 50.69 |
| A2000| densenet201 | 406 | fp16 | 45.92 |
| A2000| inceptionv3 | 120 | fp32 | 27.49 |
| A2000| inceptionv3 | 120 | fp16 | 25.27 |
| A2000| mobilenetv2 | 64 | fp32 | 3.60 |
| A2000| mobilenetv2 | 64 | fp16 | 3.24 |
| A2000| resnet18 | 31 | fp32 | 7.71 |
| A2000| resnet18 | 31 | fp16 | 6.81 |
| A2000| resnet34 | 55 | fp32 | 14.08 |
| A2000| resnet34 | 55 | fp16 | 12.02 |
| A2000| resnet50 | 72 | fp32 | 18.36 |
| A2000| resnet50 | 72 | fp16 | 14.80 |
| A2000| resnet101 | 140 | fp32 | 33.61 |
| A2000| resnet101 | 140 | fp16 | 26.66 |
| A2000| resnet152 | 208 | fp32 | 47.04 |
| A2000| resnet152 | 208 | fp16 | 37.79 |
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