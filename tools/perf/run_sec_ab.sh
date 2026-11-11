#!/bin/bash
# VKOP_SEC_CMD paired A/B at HEAD. Arms interleaved inside each cycle so thermal
# drift hits every arm equally. 208 tokens, --max-kv 320 (the canonical
# decode repro).
#   graph    -- default: packed segments, one vkQueueSubmit per segment
#   sec      -- packed segments as secondaries, one submit per round
#   rsec     -- one level per segment + replay, as secondaries
#   rsec_nb  -- rsec with the inter-segment barrier dropped
#   replay   -- segment replay WITHOUT secondaries (the 3.3x-loss baseline)
set -e
cd /Users/uqland/github/vkop
MODEL=/tmp/prof_1111/int8_fused_new.vkopbin
PROMPT='List the integers from 1 to 300, separated by commas.'
for cycle in 1 2 3; do
  for arm in graph sec rsec rsec_nb replay; do
    case "$arm" in
      graph)    extra="" ;;
      sec)      extra="VKOP_SEC_CMD=1" ;;
      rsec)     extra="VKOP_SEC_CMD=1 VKOP_SEGMENT_REPLAY=1" ;;
      rsec_nb)  extra="VKOP_SEC_CMD=1 VKOP_SEGMENT_REPLAY=1 VKOP_GRAPH_NO_BARRIER=1" ;;
      replay)   extra="VKOP_SEGMENT_REPLAY=1" ;;
    esac
    printf '%s' "$PROMPT" | env VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 $extra \
      ./build/llm_chat "$MODEL" \
      llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 208 --max-kv 320 \
      > /tmp/prof_1111/sec_${arm}_c${cycle}.out 2> /tmp/prof_1111/sec_${arm}_c${cycle}.log
    # Token stream identity is checked on the decode lines only (timings vary).
    stream=$(grep '^\[r' /tmp/prof_1111/sec_${arm}_c${cycle}.out |
             sed -E 's/^\[r[0-9]+\] [0-9.]+ms \(wall [0-9.]+\)//' | md5 -q)
    summary=$(cat /tmp/prof_1111/sec_${arm}_c${cycle}.out \
                  /tmp/prof_1111/sec_${arm}_c${cycle}.log |
              grep -E 'decode:  [0-9]+ tokens|p50 decode|p90 decode|p99 decode' |
              tr -s ' ' | tr '\n' ' ')
    echo "c$cycle $arm stream=$stream | $summary"
  done
done
