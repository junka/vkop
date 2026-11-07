#!/bin/bash
# Paired A/B/C perf run: old (84514d1 2-D int8), fix (HEAD 3-D int8), nofuse (HEAD int8, no fold).
# 3 cycles, arms interleaved within each cycle, one process per run.
set -e
cd /Users/uqland/github/vkop
PROMPT='List the integers from 1 to 300, separated by commas.'
for cycle in 1 2 3; do
  for arm in old fix nofuse; do
    case "$arm" in
      old)    model=/tmp/reg/models/int8_fused_old.vkopbin ;;
      fix)    model=/tmp/reg/models/int8_fused_fix.vkopbin ;;
      nofuse) model=/tmp/reg/models/int8_nofuse_fix.vkopbin ;;
    esac
    printf '%s' "$PROMPT" | \
      VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 VKOP_SUBMIT_PROF=1 \
      ./build/llm_chat "$model" \
      llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 208 --max-kv 320 \
      > /tmp/reg/logs/perf_${arm}_c${cycle}.log 2>&1
    echo "done $arm c$cycle"
  done
done
