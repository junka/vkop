#!/bin/bash
# Smoke test for VKOP_SEC_CMD: short decode, arms compared on the token stream.
# stdout = generated text (stderr carries the profile lines), so the .out files
# are what must match across arms.
#   graph   -- current default (packed segments, one submit per segment)
#   sec     -- packed segments executed as secondaries by one primary
#   rsec    -- one level per segment (VKOP_SEGMENT_REPLAY=1) + secondaries
set -e
cd /Users/uqland/github/vkop
MODEL=/tmp/prof_1111/int8_fused_new.vkopbin
PROMPT='List the integers from 1 to 300, separated by commas.'
for arm in graph sec rsec; do
  case "$arm" in
    graph) extra="" ;;
    sec)   extra="VKOP_SEC_CMD=1" ;;
    rsec)  extra="VKOP_SEC_CMD=1 VKOP_SEGMENT_REPLAY=1" ;;
  esac
  printf '%s' "$PROMPT" | env VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 VKOP_RUN_PROFILE=1 $extra \
    ./build/llm_chat "$MODEL" \
    llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 64 --max-kv 200 \
    > /tmp/prof_1111/smoke_${arm}.out 2> /tmp/prof_1111/smoke_${arm}.log
  echo "== $arm  out-md5=$(md5 -q /tmp/prof_1111/smoke_${arm}.out)"
  grep -E '^\[graph\]|^\[seg-replay\]|^\[runprof\]' /tmp/prof_1111/smoke_${arm}.log | tail -3
  tail -2 /tmp/prof_1111/smoke_${arm}.log | grep -E 'tok|decode' || true
done
