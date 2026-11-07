#!/bin/bash
# Decode A/B: 208 steps, 3 interleaved cycles, every MatMul/step reported.
#
# Usage: tools/perf/attribution/run_ab3.sh <arm-A.vkopbin> <arm-B.vkopbin> [reps]
#   arm names are the .vkopbin basenames; logs land in tools/perf/logs/ab3_<arm>_<cycle>.log
# Requires llm/exporter/embed_tokens.bin and llm/tokenizer/qwen3_vl.bin, and a
# llm_chat built with VKOP_SUBMIT_PROF support.
set -e
cd "$(dirname "$0")/../../.."
A=${1:?arm A model}
B=${2:?arm B model}
REPS=${3:-208}
LOGDIR=tools/perf/logs
mkdir -p "$LOGDIR"
P='List the integers from 1 to 300, separated by commas.'
for cycle in 1 2 3; do
  case $cycle in
    1) order="$A $B" ;;
    2) order="$B $A" ;;
    3) order="$A $B" ;;
  esac
  for arm in $order; do
    printf '%s' "$P" | VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 \
      VKOP_SUBMIT_PROF=1 VKOP_SUBMITPROF_TOPN=0 \
      ./build/llm_chat "$arm" llm/exporter/embed_tokens.bin \
      llm/tokenizer/qwen3_vl.bin "$REPS" --max-kv 320 \
      > "$LOGDIR/ab3_$(basename "$arm" .vkopbin)_${cycle}.log" 2>&1
    echo "done $arm cycle $cycle"
  done
done
