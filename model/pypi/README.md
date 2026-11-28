# ONNX2VKOP

Convert and optimize ONNX models to VKOP (Vulkan Compute) format for efficient inference.

## Features

- **Model Optimization**: Automatically optimize ONNX models with various fusion techniques
- **Quantization Support**: FP16 and INT8 weight-only quantization
- **DAG-based Representation**: Efficient graph representation with parallel execution support
- **Format Conversion**: NCHW to RGBA conversion for Vulkan compute shaders
- **Memory Optimization**: Unified initializer memory blocks

## Installation

```bash
pip install onnx2vkop

pip install -e .
```

## Usage
quantize with fp16

```
onnx2vkop -i model.onnx -q fp16
```

`-q fp16` produces an FP16 artifact. `onnx2vkop/fp16_graph.py` computes the
graph's FP16 domain and rewrites the activation declarations and the weight
bytes together, because the runtime picks each op's shader from the dtype of
the tensors it is handed -- an FP16 weight behind an FP32 Conv reads a 384-byte
table as a 768-byte operand, which is a wrong model, not a smaller one. Only an
op whose build follows the tensor dtype AND that has a real dual FP16 buffer
build can enter the domain; an op that cannot keeps itself and every tensor it
touches at FP32 (that verdict propagates through the graph, and Cast is the only
node allowed to sit between the two widths), and the run names the op and the
reason. An export that already declares FP16 activations but ships FP32 weights
is handled by the older per-weight pass afterwards.

Hold one layer's weights at their source precision, to localize the accuracy a
byte format cost (`onnx2vkop/exclusion.py`):

```
onnx2vkop -i model.onnx -q int4 --exclude-scope model.layers.0
onnx2vkop -i model.onnx -q int8 --exclude-node '*/attn/v_proj/MatMul' \
    --exclude-downstream-of /embed
```

Convert twice -- once with the flag, once without -- and run
`tools/vkop_vs_ort.py` on both artifacts to read which number moved; the two
runs share the input tensor and the shaders, so the delta belongs to the layer
the flag named. See the accuracy-loop section of the repository README.

### Straight from a Hugging Face checkpoint (`--hf-checkpoint`)

```
onnx2vkop --hf-checkpoint ~/models/Qwen3-4B -o qwen3_4b.vkopbin -q int4 \
    --extra-shaders TopK --master-archive build/vkop_shaders.bin
```

No separate ONNX export step and no copy of the weights. `onnx2vkop/hf_import.py`
builds the architecture from `config.json` on the meta device, traces it into a
graph-only ONNX (`<output stem>.graph.onnx`, a few MB) and points every
initializer's `external_data` straight into the checkpoint's own
`*.safetensors` shards: shard path, byte offset from the header, length. From
there it is the ordinary `-i` conversion, so every pass, `-q` format, exclusion
and audit applies, and `weight_stream` streams the weights from the shards into
the `.vkopbin` exactly as it would from an exporter's `llm.weights.bin`. Needs
`torch` and `transformers`.

- **Default builder**: Llama-family decoders (`llama`, `qwen2`, `qwen3`,
  `mistral`; tied or untied head, GQA, optional q/k norm, RoPE variants that
  only change `inv_freq` such as llama3/yarn/linear). The graph has the same
  I/O as `llm/exporter/export/qwen3_export_onnx.py` (`inputs_embeds`,
  `position_ids`, `attention_bias`, `past_key_values_i` -> `logits`,
  `present_key_values_i`), so the result runs in `llm_chat`. The embedding
  table is not in the graph; produce it with `llm/server/embed_table.py` as
  usual.
- **Other architectures**: `--hf-builder pkg.module:fn` (or `file.py:fn`),
  called as `fn(config, torch_dtype, index)` and returning a
  `hf_import.GraphSpec` (a meta-device wrapper, dummy inputs, I/O names,
  dynamic axes, plus any computed constants or name aliases).
- **Dtypes** (`--hf-trace-dtype`, default auto): the graph is traced at fp16,
  or at fp32 under `--act bf16/fp32` or for an fp32 checkpoint. A weight
  stored at that width, or stored as bf16 (the converter normalizes bf16), is
  referenced in its shard. Any other weight is converted, one tensor at a time,
  into `<stem>.graph.side.bin`.
- The shard references are relative paths that leave the ONNX's directory.
  vkop does not mind, but ONNX Runtime and `onnx.checker` refuse that. Pass
  `--hf-onnx <checkpoint>/graph.onnx` if you also want to run the graph there.
- Unsupported: dynamic-NTK RoPE (its `inv_freq` changes with the sequence
  length at run time) and Gemma's `(1 + w)` norms. Both are refused, not
  approximated.

### A per-node ONNX Runtime reference (`--emit-ort-onnx`)

The fused graph speaks vkop's own vocabulary (RMSNorm, FusedElemwise, the
rotate-half RotaryEmbedding, the vkop Attention, MatMul `swiglu`/`down_act`/
`epi_*`, Conv `activation`, ...), which no other runtime can execute, and its
fused nodes have names the source graph never had. `--emit-ort-onnx` writes
that graph lowered back to standard ONNX (`onnx2vkop/ort_lower.py`): each vkop
node becomes a plain subgraph whose last node writes the vkop node's own output
tensor, every such tensor is a graph output, and the `vkop_node_outputs`
metadata maps node names to them.

```
onnx2vkop -i model.onnx -o model.vkopbin --master-archive build/vkop_shaders.bin \
    --emit-ort-onnx model.ort.onnx
python3 tools/vkop_vs_ort.py model.vkopbin --onnx model.ort.onnx --random-fill --node-dump
```

`--node-dump` runs vkop-bench once with `VKOP_NODE_DUMP` and compares every
dumped node output with its ORT tensor in execution order, marking the first
divergence (report-only; the exit code still follows the graph outputs).
Comparing the source ONNX with `model.ort.onnx` in ORT, on the tensor names both
keep, checks the fusion passes themselves without a GPU (tests/test_ort_lower.py
does this per pass).

The cut is taken right after fusion: before the FP16 domain, weight
quantization and the weight re-layouts, so the reference is the export's own
math. Floats are computed in fp32 unless `--ort-keep-dtype` is given. The flag
forces `--stream-weights off` (the lowering needs the weight bytes), so on a
large LLM it costs the full model in RAM twice over -- prefer a trimmed or tiny
model there. Not lowered yet: LinearAttention/GatedDeltaNet, the Qwen3.5
Conv1d, EmbeddingForward, Nms, and the ring / fp8 KV cache; these raise an
error naming the node.
