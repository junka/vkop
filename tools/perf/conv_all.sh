#!/bin/bash
set -e
cd /private/tmp/vkop_old2/model/pypi
VKOP_STREAM_WEIGHTS=0 VKOP_SWIGLU_DEBUG=1 python3 -m onnx2vkop.cli \
  -i /Users/uqland/github/vkop/llm/llm.onnx \
  -o /tmp/reg/models/int8_fused_old.vkopbin -q int8 \
  --stream-weights off --master-archive /private/tmp/vkop_old2/build/vkop_shaders.bin \
  --target-backend buffer > /tmp/reg/logs/conv_old.log 2>&1
echo "OLD size=$(stat -f%z /tmp/reg/models/int8_fused_old.vkopbin) folds=$(grep -c 'Converted SwiGLU fold' /tmp/reg/logs/conv_old.log)"

cd /Users/uqland/github/vkop/model/pypi
VKOP_STREAM_WEIGHTS=0 VKOP_SWIGLU_DEBUG=1 python3 -m onnx2vkop.cli \
  -i ../../llm/llm.onnx -o /tmp/reg/models/int8_fused_fix.vkopbin -q int8 \
  --stream-weights off --master-archive ../../build/vkop_shaders.bin \
  --target-backend buffer > /tmp/reg/logs/conv_fix.log 2>&1
echo "FIX size=$(stat -f%z /tmp/reg/models/int8_fused_fix.vkopbin) folds=$(grep -c 'Converted SwiGLU fold' /tmp/reg/logs/conv_fix.log)"

VKOP_STREAM_WEIGHTS=0 VKOP_SWIGLU_DEBUG=1 python3 -m onnx2vkop.cli \
  -i ../../llm/llm.onnx -o /tmp/reg/models/int8_nofuse_fix.vkopbin -q int8 \
  --no-fusions=fuse_swiglu_down_proj --stream-weights off \
  --master-archive ../../build/vkop_shaders.bin \
  --target-backend buffer > /tmp/reg/logs/conv_nofuse.log 2>&1
echo "NOFUSE size=$(stat -f%z /tmp/reg/models/int8_nofuse_fix.vkopbin)"
