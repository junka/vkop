#!/bin/bash
# Warm m>1 prefill A/B. Two identical prompts per process: block 1 pays every
# lazily-built pipeline (MoltenVK compiles SPIR-V -> MSL at first dispatch), so
# block 5 (the second prompt's prefill) is the steady-state one to compare.
#
# Usage: tools/perf/attribution/run_pre2.sh <arm-A.vkopbin> <arm-B.vkopbin> [cycles]
set -e
cd "$(dirname "$0")/../../.."
A=${1:?arm A model}
B=${2:?arm B model}
CYCLES=${3:-3}
LOGDIR=tools/perf/logs
mkdir -p "$LOGDIR"
P='Explain the water cycle in exactly fifteen words.'
for cycle in $(seq 1 "$CYCLES"); do
  if [ $((cycle % 2)) -eq 1 ]; then order="$A $B"; else order="$B $A"; fi
  for arm in $order; do
    printf '%s\n%s\n' "$P" "$P" | VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 \
      VKOP_SUBMIT_PROF=1 VKOP_SUBMITPROF_TOPN=0 \
      ./build/llm_chat "$arm" llm/exporter/embed_tokens.bin \
      llm/tokenizer/qwen3_vl.bin 4 --max-kv 320 \
      > "$LOGDIR/pre2_$(basename "$arm" .vkopbin)_${cycle}.log" 2>&1
    echo "done $arm cycle $cycle"
  done
done
