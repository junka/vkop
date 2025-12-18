## benchmark

不会使用 CPU 做推理，CPU利用率低于2%，batch cmd 提交占用
measure 1000 rounds avgerage

| GPU | Model | Operators |Precision | latency (ms) |
| --- | ----- | --------- | -------- | ----------- |
| A2000| resnet18 | 39 | fp32 | 9.37 |
| A2000| resnet34 | 73 | fp32 | 17.10 |
| A2000| resnet50 | 90 | fp32 | 20.67 |




## 模型转换
```
python3 model/onnx2vkop.py <model.onnx>
```
