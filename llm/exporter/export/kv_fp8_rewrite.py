"""KV cache 的 fp8（E4M3）改写 —— 纯图结构变换，不读任何权重。

抽成独立模块是因为有两个调用方：qwen3vl_export_onnx.py（导出后原地改写）和
llm/exporter/make_kv_variant.py（从已经导出的 canonical llm.onnx 上离线改写，
省掉一次 3.4GB 的权重导出）。两边用的是同一份实现。
"""

from onnx import TensorProto, helper


def insert_fp8_kv_cache(model, scale=0.1):
    """Rewrite the KV cache I/O to fp8.

    For each layer i:
      - present side: the node that produces present_key_values_i now feeds a
        new QuantizeLinear node whose fp8 output is the graph output (renamed
        in place); the graph output keeps the name present_key_values_i but
        its dtype becomes float8e4m3fn.
      - past side: a new DequantizeLinear node sits between the past_key_values_i
        graph input (now fp8) and every original consumer, turning the fp8 bytes
        back into fp16 for the attention Concat.

    The past/present graph I/O dtype is recorded by the converter from the
    ValueInfoProto we set here; the runtime reads that string and creates an
    int8_t container for the fp8 bytes (llm/exporter/driver/kv_cache.hpp).
    """
    g = model.graph
    nlayers = 0
    while any(o.name == f"present_key_values_{nlayers}" for o in g.output):
        nlayers += 1
    if nlayers == 0:
        print("[kv-fp8] no present_key_values_* outputs found, skipping")
        return

    fp8_dt = TensorProto.FLOAT8E4M3FN
    # One per-tensor scale initializer, reused by every layer (same scale for
    # K and V; a single fp32 scalar).
    scale_name = "kv_cache_fp8_scale"
    scale_init = helper.make_tensor(scale_name, TensorProto.FLOAT, [], [scale])
    g.initializer.append(scale_init)

    for i in range(nlayers):
        past_name = f"past_key_values_{i}"
        pres_name = f"present_key_values_{i}"

        # ---- past side: fp8 input -> DequantizeLinear -> fp16 ------------
        past_shape = None
        for vi in list(g.input):
            if vi.name == past_name:
                past_shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in vi.type.tensor_type.shape.dim
                ]
                vi.type.tensor_type.elem_type = fp8_dt
                break
        past_fp16 = f"{past_name}_fp16"
        # Declare it as FLOAT16. Without a value_info entry, ONNX shape
        # inference types the DequantizeLinear output as float32 (DQ's default
        # with no output_dtype attribute) and propagates fp32 down the entire
        # K/V concat chain. The runtime then builds those Concat/Gather/Expand
        # tensors with fp32 containers, and the fp32 word-mover shaders
        # reinterpret the fp16 (and fp8) bytes as 32-bit words -- garbage K/V
        # and NaN attention. Forcing the bridge to fp16 makes shape inference
        # propagate fp16 to every consumer.
        g.value_info.append(helper.make_tensor_value_info(
            past_fp16, TensorProto.FLOAT16, past_shape))
        # Rewire consumers BEFORE inserting the DQ node: iterating a protobuf
        # repeated field yields fresh Python wrappers, so an `is`-identity skip
        # of the new node does NOT work and the DQ would rewrite its own input
        # into a self-loop (the converter then reports a dependency cycle).
        first_consumer = None
        for i, node in enumerate(g.node):
            for k in range(len(node.input)):
                if node.input[k] == past_name:
                    node.input[k] = past_fp16
                    if first_consumer is None:
                        first_consumer = i
        dq = helper.make_node(
            "DequantizeLinear",
            inputs=[past_name, scale_name],
            outputs=[past_fp16],
            name=f"DequantizeLinear_kv_{i}",
        )
        # Insert *before* the first node that reads `past_fp16`, not at the end.
        # The DQ depends only on a graph input and an initializer, so this
        # position is always legal — and it is the position ONNX requires:
        # appending at the end produces a graph that fails
        # onnx.checker.check_model with "Nodes in a graph must be topologically
        # sorted, however input 'past_key_values_0_fp16' of node /Gather_4 is not
        # output of any previous nodes". The converter tolerates it (it builds
        # its DAG from dependencies), so this is not a correctness fix on the
        # proven path — it is what makes the derived model checkable, and the
        # checker's shape inference is exactly what verifies the fp16 DQ bridge.
        if first_consumer is None:  # past input with no consumers at all
            g.node.append(dq)
        else:
            g.node.insert(first_consumer, dq)

        # ---- present side: fp16 producer -> QuantizeLinear -> fp8 output --
        pres_fp16 = f"{pres_name}_fp16"
        for node in g.node:
            for k in range(len(node.output)):
                if node.output[k] == pres_name:
                    node.output[k] = pres_fp16
        q = helper.make_node(
            "QuantizeLinear",
            inputs=[pres_fp16, scale_name],
            outputs=[pres_name],
            name=f"QuantizeLinear_kv_{i}",
        )
        g.node.append(q)
        for o in g.output:
            if o.name == pres_name:
                o.type.tensor_type.elem_type = fp8_dt
                break

    print(f"[kv-fp8] rewrote {nlayers} layer(s): past/present_key_values "
          f"now E4M3 fp8 (per-tensor scale={scale})")
