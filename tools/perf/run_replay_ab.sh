#!/bin/bash
# Paired A/B/C decode run at HEAD (dab1b39): graph submit without replay, with
# VKOP_SEGMENT_REPLAY=1, and the pre-graph per-level submit path.
# Arms interleaved inside each cycle; one process per run.
set -e
cd /Users/uqland/github/vkop
MODEL=/tmp/prof_1111/int8_fused_new.vkopbin
PROMPT='List the integers from 1 to 300, separated by commas.'
for cycle in 1 2 3; do
  for arm in graph replay perlevel; do
    case "$arm" in
      graph)    extra="" ;;
      replay)   extra="VKOP_SEGMENT_REPLAY=1" ;;
      perlevel) extra="VKOP_GRAPH_SUBMIT=0" ;;
    esac
    printf '%s' "$PROMPT" | env VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 $extra \
      ./build/llm_chat "$MODEL" \
      llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 208 --max-kv 320 \
      > /tmp/prof_1111/perf_${arm}_c${cycle}.log 2>&1
    echo "done $arm c$cycle"
  done
done
