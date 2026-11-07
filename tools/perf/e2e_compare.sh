#!/bin/bash
# Greedy token-stream comparison between the old 2-D int8 fold and the fixed
# 3-D int8 fold, same binary / same prompt / same max-kv.
set -e
cd /Users/uqland/github/vkop
PROMPT="${1:-What is the capital of France?}"
N="${2:-16}"
for arm in old fix; do
  printf '%s' "$PROMPT" | VKOP_NO_COOP=1 VKOP_RAW_PROMPT=1 \
    ./build/llm_chat /tmp/reg/models/int8_fused_${arm}.vkopbin \
    llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin "$N" --max-kv 64 \
    > /tmp/reg/logs/e2e_${arm}.log 2>&1
done
for arm in old fix; do
  grep -oE '→ [0-9]+' /tmp/reg/logs/e2e_${arm}.log | awk '{print $2}' | paste -sd, -
done
