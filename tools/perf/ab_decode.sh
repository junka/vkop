#!/bin/bash
# Paired decode A/B/... for llm_chat: arms interleaved inside each cycle (order
# reversed on even cycles) so clock / thermal drift hits every arm equally; one
# process per run.
#
# Usage:
#   tools/perf/ab_decode.sh [options] ARM [ARM ...]
#
#   ARM = name=model.vkopbin[,ENV=VAL ...][,BIN=path/to/llm_chat]
#     e.g.  graph=/tmp/m.vkopbin
#           replay=/tmp/m.vkopbin,VKOP_SEGMENT_REPLAY=1
#           base=/tmp/m.vkopbin,BIN=build/llm_chat_base
#
# Options:
#   -n STEPS    decode tokens                     (default 208)
#   -k MAXKV    --max-kv                          (default 320)
#   -c CYCLES   interleaved cycles                (default 3)
#   -p PROMPT   raw prompt                        (default: integers 1..300)
#   -e "ENV.."  env applied to every arm          (default: VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1)
#               add VKOP_SUBMIT_PROF=1 for per-op tables (parse_perf.py --op)
#   -o DIR      log dir                           (default tools/perf/logs)
#   -E PATH     embed table                       (default llm/exporter/embed_tokens.bin)
#   -T PATH     tokenizer                         (default llm/tokenizer/qwen3_vl.bin)
#
# Each run writes DIR/<name>_c<cycle>.log (stdout+stderr). Per run it prints an
# md5 of the token stream (decode lines with timings stripped, so arms that
# must agree numerically can be checked at a glance) and the decode summary.
# Summarise afterwards with:  tools/perf/parse_perf.py DIR/*.log
set -e
cd "$(dirname "$0")/../.."

STEPS=208 MAXKV=320 CYCLES=3 LOGDIR=tools/perf/logs
PROMPT='List the integers from 1 to 300, separated by commas.'
BASE_ENV='VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1'
EMBED=llm/exporter/embed_tokens.bin
TOK=llm/tokenizer/qwen3_vl.bin
while getopts "n:k:c:p:e:o:E:T:h" opt; do
  case $opt in
    n) STEPS=$OPTARG ;; k) MAXKV=$OPTARG ;; c) CYCLES=$OPTARG ;;
    p) PROMPT=$OPTARG ;; e) BASE_ENV=$OPTARG ;; o) LOGDIR=$OPTARG ;;
    E) EMBED=$OPTARG ;; T) TOK=$OPTARG ;;
    *) sed -n '2,30p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { sed -n '2,30p' "$0"; exit 1; }
mkdir -p "$LOGDIR"

if command -v md5 >/dev/null; then hash() { md5 -q; }; else hash() { md5sum | cut -d' ' -f1; }; fi

run_arm() {  # $1 = arm spec, $2 = cycle
  local spec=$1 cycle=$2
  local name=${spec%%=*} rest=${spec#*=}
  local model=${rest%%,*} bin=./build/llm_chat extra="" kv parts=()
  [ "$rest" != "$model" ] && IFS=, read -ra parts <<< "${rest#*,}"
  for kv in ${parts[@]+"${parts[@]}"}; do
    case $kv in BIN=*) bin=${kv#BIN=} ;; *) extra="$extra $kv" ;; esac
  done
  local log="$LOGDIR/${name}_c${cycle}.log"
  # shellcheck disable=SC2086
  printf '%s' "$PROMPT" | env $BASE_ENV $extra "$bin" "$model" "$EMBED" "$TOK" \
    "$STEPS" --max-kv "$MAXKV" > "$log" 2>&1
  local stream summary
  stream=$(grep '^\[r' "$log" | sed -E 's/^\[r[0-9]+\] [0-9.]+ms( \(wall [0-9.]+\))?//' | hash)
  summary=$(grep -E 'decode: +[0-9]+ tokens|p50 decode|p90 decode' "$log" | tr -s ' ' | tr '\n' ' ')
  echo "c$cycle $name stream=$stream | $summary"
}

arms=("$@")
for cycle in $(seq 1 "$CYCLES"); do
  if [ $((cycle % 2)) -eq 1 ]; then
    for a in "${arms[@]}"; do run_arm "$a" "$cycle"; done
  else
    for ((i = ${#arms[@]} - 1; i >= 0; i--)); do run_arm "${arms[$i]}" "$cycle"; done
  fi
done
