# cmd_probe — why segment replay's vkQueueSubmit is slow on MoltenVK

vkop's segment replay saves the per-round recording cost (~14-19ms of `vkCmd*`
for a 577-op decode round) but measured 49-180ms *inside* submit instead. This
probe asks the driver directly whether re-submitting an already-recorded
command buffer is cheaper than re-recording it, at vkop's decode round shape:
`SEG=375` segments, `DISP=2` dispatches each, one timeline semaphore per
segment, each segment waiting the previous segment's signal, batches of 32, and
a per-segment `vkWaitSemaphores` tail (what `Runtime::Run()` ends with in graph
mode).

```bash
/usr/local/bin/glslc -O0 -fshader-stage=compute nop.comp -o nop.spv
/usr/local/bin/g++-14 -std=c++17 -O2 -I/usr/local/include cmd_probe.cpp \
    -o cmd_probe -L/usr/local/lib -lvulkan
DYLD_LIBRARY_PATH=/usr/local/lib \
VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json \
    ./cmd_probe nop.spv
```

2026-11-11, AMD Radeon Pro 560X / MoltenVK, median of 37 rounds, two sessions
(ms per round):

| arm | record | submit | tailwait | total |
|---|---|---|---|---|
| re-record + SIMULTANEOUS_USE + chain | 0.24 / 0.26 | 64.7 / 31.3 | 13.2 / 6.4 | 78.9 / 38.6 |
| re-submit only + SIMULTANEOUS_USE + chain | 0.02 | 64.1 / 31.8 | 13.0 / 6.6 | 76.9 / 38.5 |
| re-submit, batch=1 | 0.01 | 59.6 | 12.3 | 72.1 |
| re-submit, **no chain** | 0.02 | 8.5 / 9.2 | 1.5 | 10.1 / 10.8 |
| re-record, **no chain** | 0.30 | 8.5 / 8.7 | 1.5 | 10.5 / 10.7 |
| re-record + ONE_TIME_SUBMIT + chain | 0.26 | 31.6 / 32.2 | 6.7 / 6.7 | 38.8 / 39.1 |

What it says:

- **re-submit is not more expensive than re-record** — the two rows are equal to
  within noise in every session. "Skipping the recording saves the host work"
  is true; "so submit stays free" is false.
- **the per-segment timeline chain is the cost**: 65ms → 8.5ms when it is
  dropped, i.e. ~0.15ms per segment submit on this driver.
- **batching the submits does not help**: `batch=1` and `batch=32` are the same,
  because the charge is per command buffer, not per `vkQueueSubmit` call.
- **SIMULTANEOUS_USE doubles the chained cost and is bimodal**: ONE_TIME sits at
  ~31.5ms in both sessions while SIMULTANEOUS reads 64.7 in one and 31.3 in the
  other — the same 3x spread the LLM replay runs show (71ms vs 219ms/token).

Absolute totals do not transfer to the LLM (the probe dispatches are empty);
the *ordering* of the columns does, and it matches `VKOP_RUN_PROFILE=1` on
llm_chat: graph mode submits 6 segments/round (0.1ms), replay mode 375
(49-180ms).

## Segment count at equal work

`nop.comp` now does `atomicAdd(buf.counter, 1u)` and the buffer is
HOST_VISIBLE|HOST_COHERENT and mapped, so every arm can prove it ran: each
round must add `SEG*DISP*256` (4 workgroups x 64 invocations x 2 dispatches),
and the arm line prints `verified` or `VERIFY-FAIL`. The arms below are all
`verified`, 37 rounds each.

Same 750 dispatches per round, only the number of command buffers changes
(ms/round):

| shape | chained submit | chained total | secondary total |
|---|---|---|---|
| 375 segments x 2 dispatches | 60.5 | 73.1 | 4.5 |
| 32 x 156 | 0.27 | 6.3 | 0.5 |
| 6 x 125 (graph's shape) | 0.07 | 1.24 | 0.45 |
| 1 x 750 | 0.02 | 0.26 | 0.25 |

The cost is per command buffer submitted, ~0.16ms on this driver: 375 of them
is 60ms, 6 of them is 0.07ms. Whether the buffer was recorded this round or
replayed from last round makes no difference.

## Secondary command buffers

`--secs` arms execute the same 375 segments as secondaries of ONE primary, with
a single timeline semaphore for the whole round and one `vkQueueSubmit`:

| arm | record | submit | tailwait | total |
|---|---|---|---|---|
| 375 primaries, chained (replay's current shape) | 0.02 | 60.6 | 12.5 | 73.1 |
| 375 secondaries, primary re-recorded per round | 0.02 | 0.48 | 3.9 | 4.5 |
| 375 secondaries, primary recorded once and only re-submitted | 0.00 | 0.61 | 3.9 | 4.6 |
| 375 secondaries split over 2 primaries, host wait between | 0.02 | 0.63 | 4.3 | 5.0 |

MoltenVK executes re-recorded secondaries on repeat submissions, at fine
segment granularity: 73ms -> 4.5ms, and the second arm does zero per-round
recording. The last arm is the shape a mid-round `copyToCPU` needs (vkop's
`pre_readback_hook` closes the segment, submits what exists, reads back, then
keeps recording) — one extra primary and its wait cost 0.4ms, so the readback
boundary is not a reason to avoid this design.

## Implemented in vkop as VKOP_SEC_CMD

`Runtime::Run()` takes the same shape when `VKOP_SEC_CMD=1`: segments are
allocated at SECONDARY level (`VulkanCommandBuffer(cmdpool, id, secondary)`),
each one ends with the `pipelineBarrier()` that orders it against the next
because `vkCmdExecuteCommands` gives no memory dependency between the buffers
it chains, and one primary per round — plus one per mid-round readback
boundary — carries them to the queue.

208-token Qwen3 int8 decode, `--max-kv 320`, `VKOP_NO_COOP=1`, 3 interleaved
cycles, token streams md5-identical in every arm (means, min-max over cycles):

| arm | env | tok/s | p50 | p90 | p99 |
|---|---|---|---|---|---|
| graph (default) | | 13.5-14.6 | 62.1-67.4 | 63.3-68.7 | 77.5-123.9 |
| sec | `VKOP_SEC_CMD=1` | 13.6-14.6 | 61.5-66.7 | 63.7-68.4 | 73.8-86.6 |
| rsec | `VKOP_SEC_CMD=1 VKOP_SEGMENT_REPLAY=1` | 18.1-18.2 | 48.6-48.7 | 49.8-50.0 | 51.9-57.2 |
| rsec_nb | + `VKOP_GRAPH_NO_BARRIER=1` | 18.0-18.1 | 48.6 | 50.1-50.2 | 53.3-62.2 |
| replay (no secondaries) | `VKOP_SEGMENT_REPLAY=1` | 4.3-4.4 | 221-223 | 226-227 | 242-265 |

`rsec` is the point of the whole exercise: the correct one-level-per-segment
split, 375/375 segments replayed, record 0.0ms and submit ~1ms per round,
+28% over graph mode. The barrier between segments is free on this driver
(`rsec` == `rsec_nb`), so it stays for portability. Script:
`tools/perf/run_sec_ab.sh`.


