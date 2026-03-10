## vkyolo

this is a simple yolo example for using vkop

- prepare yoko11 onnx from ultralytics
- convert to vkop binary ```onnx2vkop.py -i yolo11.onnx -q fp16```
- run ```vkyolo yolo11.vkopbin image.jpg```
- cross validation ```vkyolo yolo11.vkopbin image.jpg op_nodename```

run loop count should be modified in the code to 2 and
append the name of the node to cross validation, input and output of the node will be printed

## runyolo.py

this is a simple script to run yolo for a given image, used for cross validation
input:
```
--intermedia tensorname
```
this could be used to print an intermediate tensor for debugging

output:
```
[x1,y1,x2,y2,confidence,class_id]
```
this could be used for verify output of the target engine