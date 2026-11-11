#!/bin/bash
# Paired A/B at HEAD: graph submit (batch 32), segment replay (batch 32),
# segment replay with one submit per segment (VKOP_SEG_BATCH=1).
# Arms interleaved inside each cycle so clock drift cannot favour one arm.
set -e
cd /Users/uqland/github/vkop
MODEL=/tmp/prof_1111/int8_fused_new.vkopbin
PROMPT='List the integers from 1 to 300, separated by commas.'
for cycle in 1 2 3; do
  for arm in g32 r32 r1; do
    case "$arm" in
      g32) extra="VKOP_SEGMENT_REPLAY=0" ;;
      r32) extra="VKOP_SEGMENT_REPLAY=1" ;;
      r1)  extra="VKOP_SEGMENT_REPLAY=1 VKOP_SEG_BATCH=1" ;;
    esac
    printf '%s' "$PROMPT" | env VKOP_NO_COOP=1 VKOP_RAW_PROMPT=1 $extra \
      ./build/llm_chat "$MODEL" \
      llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 96 --max-kv 200 \
      > /tmp/prof_1111/ab_${arm}_c${cycle}.log 2>&1
    echo "done $arm c$cycle"
  done
done
