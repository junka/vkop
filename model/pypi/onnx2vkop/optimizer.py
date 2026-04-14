"""Model optimization utilities."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import onnx
import onnxoptimizer as optimizer
from onnx import numpy_helper, shape_inference
from onnxsim import simplify

from .dag import Node


class ONNXOptimizer:
    """Class for optimizing ONNX models."""

    @staticmethod
    def optimize_model(onnx_model, batch_size: int = 1):
        """Optimize the ONNX model using ONNX's built-in optimizer."""
        passes = [
            "eliminate_nop_cast",
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_unsqueezes",
            "fuse_consecutive_transposes",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
        ]

        initializer_names = {init.name for init in onnx_model.graph.initializer}
        actual_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initializer_names]

        input_shapes = {}
        for inp in actual_inputs:
            name = inp.name
            tensor_type = inp.type.tensor_type
            dim = tensor_type.shape.dim
            # assert len(dim) == 4, "Input shape must be 4D"
            fixed_shape = []
            for d in dim:
                if d.HasField("dim_value"):
                    fixed_shape.append(d.dim_value)
                elif d.HasField("dim_param"):
                    fixed_shape.append(batch_size)
                else:
                    fixed_shape.append(1)
            input_shapes[name] = fixed_shape

        optimized_model, check = simplify(onnx_model, overwrite_input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        optimized_model = optimizer.optimize(optimized_model, passes)
        optimized_model = shape_inference.infer_shapes(optimized_model, strict_mode=True)

        return optimized_model

    @staticmethod
    def is_topologically_sortable(graph):
        """Check if the graph is topologically sortable."""
        nodes = list(graph.node)
        n = len(nodes)

        # Step 1: 构建 tensor -> producing node index 映射
        produced_by = {}
        for idx, node in enumerate(nodes):
            for out in node.output:
                if out == "":  # 跳过空输出（虽然罕见）
                    continue
                if out in produced_by:
                    raise ValueError(f"Tensor '{out}' is produced by multiple nodes!")
                produced_by[out] = idx  # 记录生产者索引

        # 初始可用张量: inputs + initializers
        initial_tensors = {inp.name for inp in graph.input}
        initial_tensors.update(init.name for init in graph.initializer)

        # Step 2: 计算每个节点的入度（依赖的未满足输入数）
        in_degree = [0] * n
        dependents = defaultdict(list)  # idx -> list of dependent node indices

        for idx, node in enumerate(nodes):
            unmet = 0
            for inp in node.input:
                if inp == "":
                    continue
                if inp in initial_tensors:
                    continue
                if inp in produced_by:
                    producer_idx = produced_by[inp]
                    dependents[producer_idx].append(idx)
                    unmet += 1
                else:
                    raise ValueError(f"Input tensor '{inp}' is not defined in the graph!")
            in_degree[idx] = unmet

        # Step 3: Kahn's algorithm using indices
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)

        executed = 0
        while queue:
            u = queue.popleft()
            executed += 1
            for v in dependents[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if executed != n:
            raise ValueError("Graph has a cycle or unresolved dependencies!")
        return True


@dataclass
class OptimizationStats:
    """Class for storing optimization statistics."""

    nodes_before: int = 0
    nodes_after: int = 0
    initializers_before: int = 0
    initializers_after: int = 0
    rounds_executed: int = 0
    patterns_fused: Dict[str, int] = field(default_factory=dict)
    passes_applied: List[str] = field(default_factory=list)

    def report(self):
        print("=== Optimization Report ===")
        print(
            f"Node :{self.nodes_before} → {self.nodes_after} (reduce {self.nodes_before - self.nodes_after})"
        )
        if self.patterns_fused:
            print("Fused Pattern:")
            for pattern, count in self.patterns_fused.items():
                print(f"  - {pattern}: {count}")


class OptimizationPass:
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority

    def apply(self, dag_model) -> bool:
        raise NotImplementedError("Subclass must implement this method")

    def __repr__(self):
        return f"OptimizationPass({self.name}, priority={self.priority})"


class PatternBasedFusionPass(OptimizationPass):
    def __init__(self, name: str, pattern_matcher: Callable, folder: Callable, priority: int = 0):
        super().__init__(name, priority)
        self.pattern_matcher = pattern_matcher
        self.folder = folder

    def apply(self, dag_model) -> bool:
        changed = False
        matches = self.pattern_matcher(dag_model)

        for match in matches:
            if self.folder(dag_model, match):
                changed = True

        return changed


class MultiRoundOptimizer:
    def __init__(self, max_rounds: int = 10):
        self.passes: List[OptimizationPass] = []
        self.max_rounds = max_rounds
        self.stats = OptimizationStats()

    def register_pass(self, pass_instance: OptimizationPass):
        self.passes.append(pass_instance)
        self.passes.sort(key=lambda p: p.priority, reverse=True)

    def optimize(self, dag_model, verbose: bool = True) -> OptimizationStats:
        self.stats = OptimizationStats()
        self.stats.nodes_before = len(dag_model.nodes)
        self.stats.initializers_before = len(dag_model.initializers)

        if verbose:
            print(f"Multi round optimize, Initial Nodes: {self.stats.nodes_before}")
            print(f"Register {len(self.passes)} optimizer pass")

        for round_idx in range(self.max_rounds):
            round_changed = False

            if verbose:
                print(f"\n=== {round_idx + 1} round Optimization ===")

            for opt_pass in self.passes:
                try:
                    changed = opt_pass.apply(dag_model)
                    if changed:
                        round_changed = True
                        self.stats.passes_applied.append(opt_pass.name)

                        if verbose:
                            current_nodes = len(dag_model.nodes)
                            status = "✓" if changed else "○"
                            print(
                                f"  {status} {opt_pass.name} match (Current Nodes: {current_nodes})"
                            )

                except Exception as e:
                    if verbose:
                        print(f"  ✗ {opt_pass.name} Fail: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    continue

            if not round_changed:
                if verbose:
                    print(f"\nOptimization done, total {round_idx + 1} rounds")
                break

        self.stats.nodes_after = len(dag_model.nodes)
        self.stats.initializers_after = len(dag_model.initializers)

        if verbose:
            self.stats.report()

        return self.stats


class FusionOptimizer:
    """Class for fusing operators in the model."""

    @staticmethod
    def create_default_optimizer(max_rounds: int = 10) -> MultiRoundOptimizer:
        optimizer = MultiRoundOptimizer(max_rounds=max_rounds)

        optimizer.register_pass(
            PatternBasedFusionPass(
                "eliminate_identity",
                FusionOptimizer.match_identity,
                FusionOptimizer.fold_identity,
                priority=100,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "eliminate_redundant_cast",
                FusionOptimizer.match_redundant_cast,
                FusionOptimizer.fold_redundant_cast,
                priority=100,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_unsqueeze",
                FusionOptimizer.match_unsqueeze,
                FusionOptimizer.fold_unsqueeze,
                priority=100,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_squeeze",
                FusionOptimizer.match_squeeze,
                FusionOptimizer.fold_squeeze,
                priority=100,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "remove_redundant_reshape",
                FusionOptimizer.match_redundant_reshape,
                FusionOptimizer.fold_redundant_reshape,
                priority=100,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_conv_bn",
                FusionOptimizer.match_conv_bn,
                FusionOptimizer.fold_conv_bn,
                priority=90,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_add_bias_into_conv",
                FusionOptimizer.match_add_bias,
                FusionOptimizer.fold_add_bias,
                priority=88,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_gated_conv",
                FusionOptimizer.match_gated_conv,
                FusionOptimizer.fold_gated_conv,
                priority=90,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_conv_activation",
                FusionOptimizer.match_conv_activation,
                FusionOptimizer.fold_conv_activation,
                priority=85,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "replace_reducemean_reshape_with_globalaveragepool",
                FusionOptimizer.match_reducemean_reshape,
                FusionOptimizer.fold_reducemean_reshape,
                priority=75,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "unify_reduce_operators",
                FusionOptimizer.match_reduce_ops,
                FusionOptimizer.fold_reduce_ops,
                priority=75,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "replace_globalaveragepool_conv_with_gemm",
                FusionOptimizer.match_gap_conv,
                FusionOptimizer.fold_gap_conv,
                priority=75,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_isnan_where_softmax",
                FusionOptimizer.match_isnan_where_softmax,
                FusionOptimizer.fold_isnan_where_softmax,
                priority=75,
            )
        )

        optimizer.register_pass(
            PatternBasedFusionPass(
                "fuse_into_attention",
                FusionOptimizer.match_attention,
                FusionOptimizer.fold_attention,
                priority=75,
            )
        )

        return optimizer

    @staticmethod
    def optimize(dag_model, max_rounds: int = 10):
        optimizer = FusionOptimizer.create_default_optimizer(max_rounds)
        optimizer.optimize(dag_model, verbose=True)
        return dag_model

    def get_producer_consumer_from_dag(dag_model):
        """Extract producer and consumer mappings from DAG."""
        producer = {}
        consumers = defaultdict(list)

        for node in dag_model.nodes.values():
            for output in node.outputs:
                producer[output["name"]] = node

        for node in dag_model.nodes.values():
            for idx, input_tensor in enumerate(node.inputs):
                consumers[input_tensor["name"]].append((node, idx))

        return producer, consumers

    @staticmethod
    def match_identity(dag_model):
        matches = []
        for node in dag_model.nodes.values():
            if node.op_type == "Identity" and len(node.inputs) == 1 and len(node.outputs) == 1:
                matches.append({"node": node})
        return matches

    @staticmethod
    def fold_identity(dag_model, match) -> bool:
        node = match["node"]
        input_name = node.inputs[0]["name"]
        output_name = node.outputs[0]["name"]

        for other_node in dag_model.nodes.values():
            if other_node.name == node.name:
                continue
            for inp in other_node.inputs:
                if inp["name"] == output_name:
                    inp["name"] = input_name

        del dag_model.nodes[node.name]
        return True

    @staticmethod
    def match_redundant_cast(dag_model):
        matches = []

        # 匹配单个冗余Cast（输入输出类型相同）
        for node in dag_model.nodes.values():
            if node.op_type == "Cast":
                # 检查当前节点的输入类型是否与输出类型相同
                input_name = node.inputs[0]["name"]
                input_dtype = None

                if input_name in dag_model.initializers:
                    init = dag_model.initializers[input_name]
                    input_dtype = init.dtype
                else:
                    for other_node in dag_model.nodes.values():
                        for out in other_node.outputs:
                            if out["name"] == input_name:
                                input_dtype = out.get("dtype")
                                break
                        if input_dtype:
                            break
                output_dtype = node.attributes.get("to")

                if input_dtype and input_dtype == output_dtype:
                    matches.append({"node": node, "type": "redundant_single"})
                    continue

                # FIXME: 临时处理，因为onnx2vkop的Cast输入类型为None时，to为1(float32)
                data_type_map = {
                    1: "FLOAT",
                    2: "UINT8",
                    3: "INT8",
                    4: "UINT16",
                    5: "INT16",
                    6: "INT32",
                    7: "INT64",
                    8: "STRING",
                    9: "BOOL",
                    10: "FLOAT16",
                    11: "DOUBLE",
                    12: "UINT32",
                    13: "UINT64",
                    14: "COMPLEX64",
                    15: "COMPLEX128",
                    16: "BFLOAT16",
                }
                if input_dtype is None and output_dtype == 1:
                    matches.append({"node": node, "type": "redundant_single"})
                    continue

        # 匹配连续的Cast，找到可以直接删除的Cast（因为后续Cast会覆盖它的效果）
        # 构建Cast链，找到可以被跳过的Cast节点
        for node in dag_model.nodes.values():
            if node.op_type == "Cast":
                output_name = node.outputs[0]["name"]

                # 查找下一个连接的Cast节点
                for other_node in dag_model.nodes.values():
                    if other_node.op_type == "Cast":
                        for inp in other_node.inputs:
                            if inp["name"] == output_name:
                                # 检查other_node的to类型是否与node的输入类型相同
                                # 如果是这样，那么node是多余的，因为other_node直接将node的输入类型转换为目标类型
                                first_input_name = node.inputs[0]["name"]
                                first_input_dtype = None

                                # 查找第一个Cast的输入类型
                                for lookup_node in dag_model.nodes.values():
                                    for out in lookup_node.outputs:
                                        if out["name"] == first_input_name:
                                            first_input_dtype = out.get("dtype")
                                            break
                                    if first_input_dtype:
                                        break

                                second_output_dtype = other_node.attributes.get("to")

                                # 如果第一个Cast的输入类型与第二个Cast的输出类型相同，则第一个Cast是冗余的
                                if first_input_dtype and first_input_dtype == second_output_dtype:
                                    matches.append(
                                        {
                                            "redundant_node": node,  # 第一个Cast是冗余的
                                            "following_node": other_node,
                                            "type": "redundant_cast_then_cast",
                                        }
                                    )

        # 找到直接将数据转换为其自身类型的Cast（后续Cast覆盖前面的）
        all_cast_nodes = [node for node in dag_model.nodes.values() if node.op_type == "Cast"]

        # 遍历所有Cast节点，检查是否有后续Cast覆盖了它的转换
        for cast_node in all_cast_nodes:
            # 找到所有从这个Cast输出出来的路径
            output_name = cast_node.outputs[0]["name"]

            # 找到所有使用此输出的Cast节点
            connected_cast_nodes = []
            for node in all_cast_nodes:
                for inp in node.inputs:
                    if inp["name"] == output_name:
                        connected_cast_nodes.append(node)

            # 检查连接的Cast节点
            for connected_node in connected_cast_nodes:
                # 获取cast_node的输入类型
                input_name = cast_node.inputs[0]["name"]
                input_dtype = None
                for lookup_node in dag_model.nodes.values():
                    for out in lookup_node.outputs:
                        if out["name"] == input_name:
                            input_dtype = out.get("dtype")
                            break
                    if input_dtype:
                        break

                # 获取connected_node的目标类型
                target_dtype = connected_node.attributes.get("to")

                # 如果cast_node的输入类型与connected_node的目标类型相同，则cast_node是冗余的
                if input_dtype and input_dtype == target_dtype:
                    matches.append(
                        {
                            "redundant_node": cast_node,
                            "following_node": connected_node,
                            "type": "redundant_cast_then_cast",
                        }
                    )
        print(f"Matched redundant casts: {len(matches)}")
        return matches

    @staticmethod
    def fold_redundant_cast(dag_model, match) -> bool:
        match_type = match["type"]

        if match_type == "redundant_single":
            node = match["node"]
            input_name = node.inputs[0]["name"]
            output_name = node.outputs[0]["name"]

            # 将后续节点的输入从output_name重定向到input_name
            for other_node in dag_model.nodes.values():
                for inp in other_node.inputs:
                    if inp["name"] == output_name:
                        inp["name"] = input_name

            # 删除此冗余节点
            to_remove_initializers = []
            for inp in node.inputs[1:]:
                if inp["name"] in dag_model.initializers:
                    to_remove_initializers.append(inp["name"])

            del dag_model.nodes[node.name]
            for init_name in to_remove_initializers:
                if init_name in dag_model.initializers:
                    del dag_model.initializers[init_name]

            return True

        elif match_type == "redundant_chain":
            first_node = match["first_node"]
            second_node = match["second_node"]

            initial_input_name = first_node.inputs[0]["name"]
            final_output_name = second_node.outputs[0]["name"]

            # 将使用最终输出的节点重定向到初始输入
            for other_node in dag_model.nodes.values():
                for inp in other_node.inputs:
                    if inp["name"] == final_output_name:
                        inp["name"] = initial_input_name

            # 删除两个冗余的Cast节点
            del dag_model.nodes[first_node.name]
            del dag_model.nodes[second_node.name]

            return True

        return False

    @staticmethod
    def match_unsqueeze(dag_model):
        matches = []
        for node in dag_model.nodes.values():
            if node.op_type == "Unsqueeze":
                if (
                    len(node.inputs) >= 1
                    and len(node.outputs) >= 1
                    and node.inputs[0]["shape"] == node.outputs[0]["shape"]
                ):
                    matches.append({"node": node})
        return matches

    @staticmethod
    def fold_unsqueeze(dag_model, match) -> bool:
        node = match["node"]
        input_name = node.inputs[0]["name"]
        output_name = node.outputs[0]["name"]

        for other_node in dag_model.nodes.values():
            if other_node.name == node.name:
                continue
            for inp in other_node.inputs:
                if inp["name"] == output_name:
                    inp["name"] = input_name

        del dag_model.nodes[node.name]
        return True

    @staticmethod
    def match_squeeze(dag_model):
        matches = []
        for node in dag_model.nodes.values():
            if node.op_type == "Squeeze":
                if (
                    len(node.inputs) >= 1
                    and len(node.outputs) >= 1
                    and node.inputs[0]["shape"] == node.outputs[0]["shape"]
                ):
                    matches.append({"node": node})
        return matches

    @staticmethod
    def fold_squeeze(dag_model, match) -> bool:
        node = match["node"]
        input_name = node.inputs[0]["name"]
        output_name = node.outputs[0]["name"]

        for other_node in dag_model.nodes.values():
            if other_node.name == node.name:
                continue
            for inp in other_node.inputs:
                if inp["name"] == output_name:
                    inp["name"] = input_name

        del dag_model.nodes[node.name]
        return True

    @staticmethod
    def match_redundant_reshape(dag_model):
        matches = []
        for node in dag_model.nodes.values():
            if node.op_type == "Reshape":
                if (
                    len(node.inputs) >= 1
                    and len(node.outputs) >= 1
                    and node.inputs[0]["shape"] == node.outputs[0]["shape"]
                ):
                    matches.append({"node": node})
        return matches

    @staticmethod
    def fold_redundant_reshape(dag_model, match) -> bool:
        node = match["node"]
        input_name = node.inputs[0]["name"]
        output_name = node.outputs[0]["name"]

        for other_node in dag_model.nodes.values():
            if other_node.name == node.name:
                continue
            for inp in other_node.inputs:
                if inp["name"] == output_name:
                    inp["name"] = input_name

        to_remove_initializers = []
        for inp in node.inputs[1:]:
            if inp["name"] in dag_model.initializers:
                to_remove_initializers.append(inp["name"])

        del dag_model.nodes[node.name]
        for init_name in to_remove_initializers:
            del dag_model.initializers[init_name]

        return True

    @staticmethod
    def match_conv_bn(dag_model):
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        for node in dag_model.nodes.values():
            if node.op_type != "Conv":
                continue

            conv_out = node.outputs[0]["name"]
            conv_consumers = consumers.get(conv_out, [])

            if len(conv_consumers) != 1:
                continue

            bn_node, _ = conv_consumers[0]
            if bn_node.op_type != "BatchNormalization":
                continue

            matches.append(
                {
                    "conv_node": node,
                    "bn_node": bn_node,
                }
            )

        return matches

    @staticmethod
    def fold_conv_bn(dag_model, match) -> bool:
        node = match["conv_node"]
        bn_node = match["bn_node"]

        # inputs[1] are merged scale, bias, mean, variance respectively
        tensor_name = bn_node.inputs[1]["name"]
        if tensor_name not in dag_model.initializers:
            print(
                f"Warning: BatchNormalization parameters {tensor_name} not found, skipping fusion"
            )
            return False

        tensor_array = numpy_helper.to_array(dag_model.initializers[tensor_name])
        total_elements = len(tensor_array)
        padded_N = total_elements // 4

        scale_array = np.zeros(padded_N, dtype=np.float32)
        bias_array = np.zeros(padded_N, dtype=np.float32)
        mean_array = np.zeros(padded_N, dtype=np.float32)
        var_array = np.zeros(padded_N, dtype=np.float32)
        for i in range(padded_N // 4):
            base_idx = i * 16
            for j in range(4):
                scale_array[i * 4 + j] = tensor_array[base_idx + j]  # scale
                bias_array[i * 4 + j] = tensor_array[base_idx + 4 + j]  # bias
                mean_array[i * 4 + j] = tensor_array[base_idx + 8 + j]  # mean
                var_array[i * 4 + j] = tensor_array[base_idx + 12 + j]

        eps = float(bn_node.attributes.get("epsilon", 1e-5))

        has_conv_bias = len(node.inputs) > 2  # 输入0是data, 输入1是weights, 输入2是bias(如果有)

        conv_weight_name = node.inputs[1]["name"]
        if conv_weight_name not in dag_model.initializers:
            print(f"Warning: Conv weight {conv_weight_name} not found, skipping fusion")
            return False

        conv_weight = numpy_helper.to_array(dag_model.initializers[conv_weight_name])

        if has_conv_bias:
            conv_bias_name = node.inputs[2]["name"]
            if conv_bias_name not in dag_model.initializers:
                print(f"Warning: Conv bias {conv_bias_name} not found, skipping fusion")
                return False
            conv_bias = numpy_helper.to_array(dag_model.initializers[conv_bias_name])
        else:
            conv_bias = np.zeros(scale_array.shape, dtype=scale_array.dtype)

        print("Fusing Conv node", node.name, "with BN node", bn_node.name)
        # 执行参数融合
        # 计算: gamma / sqrt(var + eps)
        inv_std = scale_array / np.sqrt(var_array + eps)

        # 新权重 = 旧权重 * (gamma / sqrt(var + eps))
        # 对于卷积，我们需要正确处理维度
        fused_weight = conv_weight * inv_std.reshape(
            -1, 1, 1, 1
        )  # reshape to match conv weight shape

        # 新偏置 = (旧偏置 - mean) * (gamma / sqrt(var + eps)) + beta
        fused_bias = (conv_bias - mean_array) * inv_std + bias_array

        # 直接更新dag_model.initializers中的权重和偏置
        # 更新权重为融合后的权重
        fused_weight_tensor = numpy_helper.from_array(fused_weight, conv_weight_name)
        dag_model.initializers[conv_weight_name] = fused_weight_tensor

        # 为偏置创建名称（如果原来没有bias，现在添加）
        if has_conv_bias:
            conv_bias_name = node.inputs[2]["name"]
        else:
            conv_bias_name = f"{node.name}_fused_bias"

        fused_bias_tensor = numpy_helper.from_array(fused_bias, conv_bias_name)
        dag_model.initializers[conv_bias_name] = fused_bias_tensor

        fused = Node(
            op_type=node.op_type,
            name=node.name,
            attributes=dict(node.attributes),
            inputs=[
                node.inputs[0],  # 输入数据
                node.inputs[1],  # 权重（名称不变，但数值已更新）
                {"name": conv_bias_name, "shape": list(fused_bias.shape)},
            ],
            outputs=bn_node.outputs,
        )

        del dag_model.nodes[node.name]
        del dag_model.nodes[bn_node.name]
        dag_model.nodes[fused.name] = fused

        print("Fusing Conv+BN patterns...")
        return True

    @staticmethod
    def match_gated_conv(dag_model):
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        for node in dag_model.nodes.values():
            if node.op_type != "Mul" or len(node.inputs) != 2:
                continue

            inp0, inp1 = node.inputs[0]["name"], node.inputs[1]["name"]
            candidates = [(inp0, inp1), (inp1, inp0)]

            matched = False
            for conv_out, sig_out in candidates:
                if conv_out not in producer or sig_out not in producer:
                    continue

                conv_node = producer[conv_out]
                sig_node = producer[sig_out]

                if conv_node.op_type != "Conv" or sig_node.op_type != "Sigmoid":
                    continue

                if len(sig_node.inputs) != 1:
                    continue

                if sig_node.inputs[0]["name"] != conv_out:
                    continue

                conv_consumers = consumers[conv_out]
                if len(conv_consumers) != 2:
                    continue

                consumer_names = {c[0].op_type for c in conv_consumers}
                if consumer_names != {"Sigmoid", "Mul"}:
                    continue

                matches.append(
                    {
                        "conv_node": conv_node,
                        "sig_node": sig_node,
                        "mul_node": node,
                    }
                )
                matched = True
                break

            if matched:
                continue

        return matches

    @staticmethod
    def fold_gated_conv(dag_model, match) -> bool:
        conv_node = match["conv_node"]
        sig_node = match["sig_node"]
        mul_node = match["mul_node"]

        fused = Node(
            op_type=conv_node.op_type,
            name=conv_node.name,
            attributes=dict(conv_node.attributes),
            inputs=conv_node.inputs,
            outputs=mul_node.outputs,
        )
        fused.attributes["activation"] = "Swish"

        del dag_model.nodes[conv_node.name]
        del dag_model.nodes[sig_node.name]
        del dag_model.nodes[mul_node.name]
        dag_model.nodes[fused.name] = fused

        return True

    @staticmethod
    def match_conv_activation(dag_model):
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)
        ACTIVATIONS = {"Relu", "Sigmoid", "Tanh", "HardSwish", "Mish"}

        for node in dag_model.nodes.values():
            if node.op_type not in ["Conv", "BatchNormalization", "Add", "Gemm"]:
                continue

            if len(node.outputs) != 1:
                continue

            out_name = node.outputs[0]["name"]
            outs = consumers.get(out_name, [])

            if len(outs) != 1:
                continue

            next_node, _ = outs[0]

            if next_node.op_type not in ACTIVATIONS:
                continue

            if len(next_node.inputs) != 1:
                continue

            if next_node.inputs[0]["name"] != out_name:
                continue

            matches.append(
                {
                    "base_node": node,
                    "activation_node": next_node,
                }
            )

        return matches

    @staticmethod
    def fold_conv_activation(dag_model, match) -> bool:
        base_node = match["base_node"]
        activation_node = match["activation_node"]

        fused = Node(
            op_type=base_node.op_type,
            name=base_node.name,
            attributes=dict(base_node.attributes),
            inputs=base_node.inputs,
            outputs=activation_node.outputs,
        )
        fused.attributes["activation"] = activation_node.op_type

        del dag_model.nodes[base_node.name]
        del dag_model.nodes[activation_node.name]
        dag_model.nodes[fused.name] = fused

        return True

    @staticmethod
    def match_gap_conv(dag_model):
        """
        Match Conv that follows GlobalAveragePool for GEMM replacement.

        Conditions:
        1. GlobalAveragePool output shape is [N, C, 1, 1]
        2. Conv has kernel size 1x1 and stride 1x1
        3. Conv has no padding or padding that maintains the 1x1 output
        """
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        for node in dag_model.nodes.values():
            if node.op_type != "Conv":
                continue

            conv_input_name = node.inputs[0]["name"]

            if conv_input_name not in producer:
                continue

            gap_node = producer[conv_input_name]

            if gap_node.op_type != "GlobalAveragePool":
                continue

            kernel_shape = node.attributes.get("kernel_shape", [1, 1])
            strides = node.attributes.get("strides", [1, 1])
            pads = node.attributes.get("pads", [0, 0, 0, 0])

            if kernel_shape == [1, 1] and strides == [1, 1]:
                if pads == [0, 0, 0, 0] or (
                    pads[0] == pads[2] and pads[1] == pads[3] and pads[0] <= 1 and pads[1] <= 1
                ):
                    weight_name = node.inputs[1]["name"]
                    weight_tensor = None

                    if weight_name in dag_model.initializers:
                        weight_tensor = dag_model.initializers[weight_name]
                        original_shape = list(weight_tensor.dims)

                        if not (
                            len(original_shape) == 4
                            and original_shape[2] == 1
                            and original_shape[3] == 1
                        ):
                            continue

                    matches.append(
                        {
                            "gap_node": gap_node,
                            "conv_node": node,
                            "weight_tensor": weight_tensor,
                        }
                    )

        return matches

    @staticmethod
    def fold_gap_conv(dag_model, match) -> bool:
        """Fold matched GAP + Conv pattern into GEMM."""
        gap_node = match["gap_node"]
        conv_node = match["conv_node"]
        weight_tensor = match["weight_tensor"]

        weight_name = conv_node.inputs[1]["name"]

        if weight_tensor and weight_name in dag_model.initializers:
            original_shape = list(weight_tensor.dims)
            new_shape = [original_shape[0], original_shape[1]]

            weight_data = numpy_helper.to_array(weight_tensor)
            reshaped_weight = weight_data.reshape(new_shape)

            new_weight_tensor = numpy_helper.from_array(reshaped_weight, weight_name)
            new_weight_tensor.data_type = weight_tensor.data_type

            dag_model.initializers[weight_name] = new_weight_tensor

        gemm_attributes = {
            "alpha": 1.0,
            "beta": 1.0,
            "transA": 0,
            "transB": 0,
        }

        gemm_inputs = [
            gap_node.outputs[0],
            {
                "name": weight_name,
                "shape": (
                    [original_shape[0], original_shape[1]]
                    if weight_tensor
                    else conv_node.inputs[1]["shape"][:2]
                ),
            },
        ]

        if len(conv_node.inputs) > 2:
            gemm_inputs.append(conv_node.inputs[2])

        gemm_outputs = []
        for output in conv_node.outputs:
            orig_shape = output["shape"]
            new_shape = orig_shape[:]
            if len(new_shape) >= 2 and new_shape[-2:] == [1, 1]:
                new_shape = new_shape[:-2]

            gemm_outputs.append({"name": output["name"], "shape": new_shape})

            if hasattr(dag_model, "outputs"):
                for model_output in dag_model.outputs:
                    if model_output["name"] == output["name"]:
                        model_output["shape"] = new_shape

        gemm_node = Node(
            op_type="Gemm",
            name=f"Gemm_after_GAP_{conv_node.name}",
            attributes=gemm_attributes,
            inputs=gemm_inputs,
            outputs=gemm_outputs,
        )

        del dag_model.nodes[conv_node.name]
        dag_model.nodes[gemm_node.name] = gemm_node

        print(f"Replaced Conv '{conv_node.name}' after GlobalAveragePool with GEMM")
        return True

    @staticmethod
    def match_reducemean_reshape(dag_model):
        """
        Match ReduceMean + Reshape pattern for GlobalAveragePool replacement.

        Conditions:
        1. ReduceMean operates on HW dimensions (axes=[2, 3] or [-2, -1])
        2. Either:
           - Reshape removes trailing 1x1 dimensions (keepdims=1), OR
           - No reshape needed (keepdims=0 produces desired output shape)
        """
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        for node in dag_model.nodes.values():
            if node.op_type != "ReduceMean":
                continue

            axes = []
            keepdims = node.attributes.get("keepdims", 1)
            noop_with_empty_axes = node.attributes.get("noop_with_empty_axes", 0)

            if "axes" in node.attributes:
                axes = node.attributes["axes"]
            else:
                if len(node.inputs) > 1:
                    axes_input_name = node.inputs[1]["name"]
                    if axes_input_name in dag_model.initializers:
                        axes_tensor = dag_model.initializers[axes_input_name]
                        axes_array = numpy_helper.to_array(axes_tensor)
                        axes = (
                            axes_array.tolist()
                            if hasattr(axes_array, "tolist")
                            else list(axes_array)
                        )

            if len(axes) == 0 and noop_with_empty_axes == 1:
                continue
            elif len(axes) == 0 and noop_with_empty_axes == 0:
                continue

            input_shape = node.inputs[0]["shape"]
            normalized_axes = []
            for ax in axes:
                if ax < 0:
                    normalized_axes.append(len(input_shape) + ax)
                else:
                    normalized_axes.append(ax)

            if sorted(normalized_axes) != [2, 3]:
                continue

            output_after_reduce = node.outputs[0]["shape"]

            if keepdims == 1:
                reducemean_output_name = node.outputs[0]["name"]
                reducemean_consumers = consumers.get(reducemean_output_name, [])

                if len(reducemean_consumers) != 1:
                    continue

                reshape_node, _ = reducemean_consumers[0]

                if reshape_node.op_type != "Reshape":
                    continue

                reshape_node = dag_model.nodes.get(reshape_node.name)
                if not reshape_node:
                    continue

                output_after_reshape = reshape_node.outputs[0]["shape"]

                if (
                    len(output_after_reduce) == 4
                    and output_after_reduce[2] == 1
                    and output_after_reduce[3] == 1
                    and len(output_after_reshape) == len(input_shape) - 2
                    and output_after_reshape[:2] == output_after_reduce[:2]
                ):
                    matches.append(
                        {
                            "reducemean_node": node,
                            "reshape_node": reshape_node,
                            "case": "keepdims_1",
                        }
                    )

            elif keepdims == 0:
                if (
                    len(input_shape) == 4
                    and len(output_after_reduce) == 2
                    and input_shape[0] == output_after_reduce[0]
                    and input_shape[1] == output_after_reduce[1]
                ):
                    matches.append(
                        {
                            "reducemean_node": node,
                            "reshape_node": None,
                            "case": "keepdims_0",
                        }
                    )

        return matches

    @staticmethod
    def fold_reducemean_reshape(dag_model, match) -> bool:
        """Fold matched ReduceMean + Reshape pattern into GlobalAveragePool."""
        reducemean_node = match["reducemean_node"]
        reshape_node = match["reshape_node"]
        case = match["case"]

        globalavgpool_node = Node(
            op_type="GlobalAveragePool",
            name=f"GlobalAveragePool_from_{reducemean_node.name}"
            + (f"_to_{reshape_node.name}" if reshape_node else "_keepdims0"),
            attributes={},
            inputs=[reducemean_node.inputs[0]],
            outputs=[],
        )

        if case == "keepdims_1" and reshape_node:
            globalavgpool_node.outputs = [
                {
                    "name": reshape_node.outputs[0]["name"],
                    "shape": reshape_node.outputs[0]["shape"],
                }
            ]
            del dag_model.nodes[reshape_node.name]
        elif case == "keepdims_0":
            globalavgpool_node.outputs = [reducemean_node.outputs[0]]

        del dag_model.nodes[reducemean_node.name]
        dag_model.nodes[globalavgpool_node.name] = globalavgpool_node

        print(
            f"Replaced ReduceMean '{reducemean_node.name}'"
            + (f" and Reshape '{reshape_node.name}'" if reshape_node else "")
            + " with GlobalAveragePool"
        )
        return True

    @staticmethod
    def match_reduce_ops(dag_model):
        """Match all reduceXX operators for unification."""
        matches = []

        reduce_ops_map = {
            "ReduceSum": "sum",
            "ReduceMean": "mean",
            "ReduceMax": "max",
            "ReduceMin": "min",
            "ReduceProd": "prod",
            "ReduceSumSquare": "sum_square",
            "ReduceL1": "l1_norm",
            "ReduceL2": "l2_norm",
            "ReduceLogSum": "log_sum",
            "ReduceLogSumExp": "log_sum_exp",
        }

        for node in dag_model.nodes.values():
            op_type = node.op_type

            if op_type not in reduce_ops_map:
                continue

            axes = []
            noop_with_empty_axes = node.attributes.get("noop_with_empty_axes", 0)

            if "axes" in node.attributes:
                axes = node.attributes["axes"]
            else:
                if len(node.inputs) > 1:
                    axes_input_name = node.inputs[1]["name"]
                    if axes_input_name in dag_model.initializers:
                        axes_tensor = dag_model.initializers[axes_input_name]
                        axes_array = numpy_helper.to_array(axes_tensor)
                        axes = (
                            axes_array.tolist()
                            if hasattr(axes_array, "tolist")
                            else list(axes_array)
                        )

            if len(axes) == 0:
                if noop_with_empty_axes == 1:
                    continue
                elif noop_with_empty_axes == 0:
                    input_shape = node.inputs[0]["shape"]
                    axes = list(range(len(input_shape)))

            input_shape = node.inputs[0]["shape"]
            normalized_axes = []
            for ax in axes:
                if ax < 0:
                    normalized_axes.append(len(input_shape) + ax)
                else:
                    normalized_axes.append(ax)

            matches.append(
                {
                    "node": node,
                    "reduce_op": reduce_ops_map[op_type],
                    "normalized_axes": normalized_axes,
                }
            )

        return matches

    @staticmethod
    def fold_reduce_ops(dag_model, match) -> bool:
        """Fold matched reduce operator into unified Reduce node."""
        node = match["node"]
        reduce_op = match["reduce_op"]
        normalized_axes = match["normalized_axes"]

        reduce_attributes = {
            "reduce_op": reduce_op,
            "axes": normalized_axes,
            "keepdims": node.attributes.get("keepdims", 1),
        }

        reduce_inputs = [node.inputs[0]]

        reduce_node = Node(
            op_type="Reduce",
            name=node.name,
            attributes=reduce_attributes,
            inputs=reduce_inputs,
            outputs=node.outputs[:],
        )

        del dag_model.nodes[node.name]
        dag_model.nodes[reduce_node.name] = reduce_node

        print(f"Unified {node.op_type} '{node.name}' into Reduce node")
        return True

    @staticmethod
    def match_add_bias(dag_model):
        """
        Match Conv + Add pattern where Add adds bias to Conv output.

        Conditions:
        1. Conv has no bias (only 2 inputs: data and weights)
        2. Add's second input is a 1D constant tensor (bias)
        3. Add has only one consumer or the Conv output is only used by Add
        """
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        for node in dag_model.nodes.values():
            if node.op_type != "Conv":
                continue

            if len(node.inputs) != 2:
                continue

            conv_out_name = node.outputs[0]["name"]
            conv_consumers = consumers.get(conv_out_name, [])

            if len(conv_consumers) != 1:
                continue

            add_node, _ = conv_consumers[0]

            if add_node.op_type != "Add":
                continue

            if len(add_node.inputs) != 2:
                continue

            bias_input = None
            for inp in add_node.inputs:
                if inp["name"] != conv_out_name:
                    bias_input = inp
                    break

            if not bias_input:
                continue

            bias_name = bias_input["name"]
            if bias_name not in dag_model.initializers:
                continue

            bias_tensor = dag_model.initializers[bias_name]
            bias_array = numpy_helper.to_array(bias_tensor)

            if len(bias_array.shape) != 1:
                continue

            matches.append(
                {
                    "conv_node": node,
                    "add_node": add_node,
                    "bias_name": bias_name,
                    "bias_array": bias_array,
                }
            )

        return matches

    @staticmethod
    def fold_add_bias(dag_model, match) -> bool:
        """Fold Conv + Add(bias) into Conv with bias."""
        conv_node = match["conv_node"]
        add_node = match["add_node"]
        # bias_name = match["bias_name"]
        bias_array = match["bias_array"]

        new_bias_name = f"{conv_node.name}_bias"
        bias_tensor = numpy_helper.from_array(bias_array, new_bias_name)
        dag_model.initializers[new_bias_name] = bias_tensor

        fused = Node(
            op_type=conv_node.op_type,
            name=conv_node.name,
            attributes=dict(conv_node.attributes),
            inputs=[
                conv_node.inputs[0],
                conv_node.inputs[1],
                {"name": new_bias_name, "shape": list(bias_array.shape)},
            ],
            outputs=add_node.outputs,
        )

        del dag_model.nodes[conv_node.name]
        del dag_model.nodes[add_node.name]
        dag_model.nodes[fused.name] = fused

        print(f"Fused Conv '{conv_node.name}' + Add bias into Conv with bias")
        return True

    @staticmethod
    def match_isnan_where_softmax(dag_model):
        """
        Match pattern where IsNaN + Where is used to create a mask for Softmax.

        The pattern to match:
            softmax
            |     |
        IsNaN     |
            |     |
            Where
            |
        """
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        # 遍历所有Softmax节点
        for softmax_node in dag_model.nodes.values():
            if softmax_node.op_type != "Softmax":
                continue

            # 检查Softmax的输出是否同时连接到IsNaN和Where节点
            softmax_output_name = softmax_node.outputs[0]["name"]
            
            # 获取使用这个输出的所有消费者节点
            softmax_consumers = consumers.get(softmax_output_name, [])
            
            # 分离IsNaN和Where节点
            isnan_node = None
            where_node = None
            
            for consumer_node, _ in softmax_consumers:
                if consumer_node.op_type == "IsNaN":
                    isnan_node = consumer_node
                elif consumer_node.op_type == "Where":
                    where_node = consumer_node
            
            # 如果同时找到了IsNaN和Where节点
            if isnan_node and where_node:
                # 检查IsNaN的输出是否也连接到Where节点
                isnan_output_name = isnan_node.outputs[0]["name"]
                isnan_consumers = consumers.get(isnan_output_name, [])
                
                # 检查Where节点是否消费了IsNaN的输出
                isnan_output_used_by_where = any(
                    consumer_node == where_node for consumer_node, _ in isnan_consumers
                )
                
                if not isnan_output_used_by_where:
                    continue
                    
                # 验证Where节点有三个输入：condition, X, Y
                if len(where_node.inputs) == 3:
                    # 找到三个输入：条件、X值、Y值
                    condition_input = None
                    x_input = None
                    y_input = None

                    for i, inp in enumerate(where_node.inputs):
                        if inp["name"] == isnan_output_name:  # IsNaN的输出
                            condition_input = inp
                        elif inp["name"] == softmax_output_name:  # Softmax的输出
                            if x_input is None:
                                x_input = inp
                            else:
                                y_input = inp
                        else:  # 其他输入（通常是常量）
                            if x_input is None:
                                x_input = inp
                            elif y_input is None:
                                y_input = inp

                    # 确保找到了所有输入
                    if condition_input and x_input and y_input:
                        matches.append(
                            {
                                "softmax_node": softmax_node,
                                "isnan_node": isnan_node,
                                "where_node": where_node,
                                "condition_input": condition_input,
                                "x_input": x_input,
                                "y_input": y_input,
                            }
                        )
        return matches

    @staticmethod
    def fold_isnan_where_softmax(dag_model, match) -> bool:
        """
        Fold the IsNaN + Where pattern after Softmax into a modified Softmax operator.
        """
        softmax_node = match["softmax_node"]
        isnan_node = match["isnan_node"]
        where_node = match["where_node"]

        # 检查Where的Y输入是否是常量（通常是0）
        y_input = match["y_input"]
        replacement_value = 0.0

        if y_input["name"] in dag_model.initializers:
            # 如果Y输入是初始化器（常量），获取其值
            initializer = dag_model.initializers[y_input["name"]]
            tensor_array = numpy_helper.to_array(initializer)
            replacement_value = tensor_array.item() if tensor_array.size == 1 else 0.0

        # 获取Softmax的属性
        softmax_attrs = dict(softmax_node.attributes)
        softmax_attrs["nan_optimization"] = True
        softmax_attrs["nan_replacement_value"] = float(replacement_value)

        # 创建融合后的Softmax节点
        fused_softmax_node = Node(
            op_type="Softmax",
            name=f"{softmax_node.name}_nan_handled",
            attributes=softmax_attrs,
            inputs=softmax_node.inputs[:],  # 使用原始Softmax的输入
            outputs=where_node.outputs[:],  # 输出使用Where的输出
        )

        # 删除旧的节点
        nodes_to_delete = [isnan_node, where_node]
        for node in nodes_to_delete:
            if node.name in dag_model.nodes:
                del dag_model.nodes[node.name]

        # 替换Softmax节点
        if softmax_node.name in dag_model.nodes:
            del dag_model.nodes[softmax_node.name]

        # 添加融合后的节点
        dag_model.nodes[fused_softmax_node.name] = fused_softmax_node

        print(
            f"Fused Softmax + IsNaN + Where pattern into enhanced Softmax: {fused_softmax_node.name}"
        )
        return True

    @staticmethod
    def match_attention(dag_model):
        """
        Match attention patterns for potential fusion.

        The pattern to match:
            Q          K             V
            |          |             |
        MUL(scale)    MUL(scale)     |
            |          |             |
            |       Transpose        |
            |          |             |
            ---MatMul---             |
                |                    |
        att_mask---Add               |
                |                    |
                |                    |
            Softmax                  |
                |                    |
                -----MatMul----------
                        |
                        Y
        """
        matches = []
        producer, consumers = FusionOptimizer.get_producer_consumer_from_dag(dag_model)

        # 首先寻找Softmax节点，这是Attention模式的关键特征之一
        for softmax_node in dag_model.nodes.values():
            if softmax_node.op_type != "Softmax":
                continue

            softmax_input_name = softmax_node.inputs[0]["name"]
            input_producer = producer.get(softmax_input_name)

            add_node = None
            att_mask_input = None
            matmul_qk_node = None

            if input_producer and input_producer.op_type == "Add":
                add_node = input_producer
                # Add节点应该有两个输入：一个来自matmul_qk，一个来自att_mask
                for inp in add_node.inputs:
                    inp_producer = producer.get(inp["name"])
                    if inp_producer and inp_producer.op_type == "MatMul":
                        matmul_qk_node = inp_producer
                    elif inp["name"] != softmax_input_name:  # 另一个输入是att_mask
                        att_mask_input = inp

                # 如果Add节点没有MatMul输入，则不是我们要找的模式
                if not matmul_qk_node:
                    continue
            elif input_producer and input_producer.op_type == "MatMul":
                # 没有Add节点，直接就是MatMul -> Softmax
                matmul_qk_node = input_producer
            else:
                # 既不是Add也不是MatMul，不符合模式
                continue

            if not matmul_qk_node:
                continue

            print(f"Found Softmax node '{softmax_node.name}' with preceding MatMul '{matmul_qk_node.name}' and Add '{add_node.name if add_node else 'None'}'")

            input_paths = []
            for inp in matmul_qk_node.inputs:
                inp_name = inp["name"]
                inp_producer = producer.get(inp_name)
                
                path_info = {
                    "input_name": inp_name,
                    "producer": inp_producer,
                    "is_scale_only": False,      # 只有scale操作 (Mul/Div)
                    "is_transpose_only": False,  # 只有transpose操作
                    "has_scale_and_transpose": False,  # 同时有scale和transpose操作（顺序无关）
                    "original_node": None
                }
                
                if inp_producer:
                    if inp_producer.op_type == "Transpose":
                        # 检查Transpose的输入是否有scale操作 (Transpose -> Scale)
                        transpose_input_name = inp_producer.inputs[0]["name"]
                        transpose_input_producer = producer.get(transpose_input_name)
                        
                        if transpose_input_producer and transpose_input_producer.op_type in ["Mul", "Div"]:
                            # Transpose <- Scale: 有scale和transpose操作
                            path_info["has_scale_and_transpose"] = True
                            # 查找原始节点
                            for scale_inp in transpose_input_producer.inputs:
                                if scale_inp["name"] != transpose_input_name and scale_inp["name"] in dag_model.initializers:
                                    path_info["original_node"] = producer.get(
                                        transpose_input_producer.inputs[0]["name"]
                                        if transpose_input_producer.inputs[0]["name"] != scale_inp["name"]
                                        else transpose_input_producer.inputs[1]["name"]
                                    )
                                    break
                            if path_info["original_node"] is None:
                                path_info["original_node"] = transpose_input_producer
                        else:
                            # Transpose only
                            path_info["is_transpose_only"] = True
                            path_info["original_node"] = transpose_input_producer
                            
                    elif inp_producer.op_type in ["Mul", "Div"]:
                        # 检查是否有initializer作为scale因子
                        has_initializer_scale = False
                        for scale_inp in inp_producer.inputs:
                            if scale_inp["name"] != inp_name and scale_inp["name"] in dag_model.initializers:
                                has_initializer_scale = True
                                break
                        
                        if has_initializer_scale:
                            # 检查Scale的输入是否有Transpose (Scale <- Transpose)
                            scale_input_name = None
                            for scale_inp in inp_producer.inputs:
                                if scale_inp["name"] != inp_name and scale_inp["name"] in dag_model.initializers:
                                    continue  # 这是scale值，不是数据输入
                                else:
                                    scale_input_name = scale_inp["name"]
                                    break
                            
                            if scale_input_name:
                                scale_input_producer = producer.get(scale_input_name)
                                if scale_input_producer and scale_input_producer.op_type == "Transpose":
                                    # Scale <- Transpose: 有scale和transpose操作（顺序相反）
                                    path_info["has_scale_and_transpose"] = True
                                    # 获取Transpose的输入作为原始节点
                                    transpose_input_name = scale_input_producer.inputs[0]["name"]
                                    path_info["original_node"] = producer.get(transpose_input_name)
                                else:
                                    # Scale only
                                    path_info["is_scale_only"] = True
                                    # 查找原始节点
                                    for scale_inp in inp_producer.inputs:
                                        if scale_inp["name"] != inp_name and scale_inp["name"] in dag_model.initializers:
                                            path_info["original_node"] = producer.get(
                                                inp_producer.inputs[0]["name"]
                                                if inp_producer.inputs[0]["name"] != scale_inp["name"]
                                                else inp_producer.inputs[1]["name"]
                                            )
                                            break
                        else:
                            # Scale without initializer: 可能是其他类型的scale
                            path_info["original_node"] = inp_producer
                    else:
                        # 其他类型：可能是原始的Q或K
                        path_info["original_node"] = inp_producer
                
                input_paths.append(path_info)

            # 需要找到一个scale-only路径和一个scale+transpose或transpose-only路径
            scale_only_path = None
            scale_transpose_path = None
            
            for path in input_paths:
                if path["is_scale_only"] and path["original_node"]:
                    scale_only_path = path
                elif (path["has_scale_and_transpose"]) and path["original_node"]:
                    scale_transpose_path = path

            if not scale_only_path or not scale_transpose_path:
                continue

            # 3. 现在我们有了Q和K路径，继续检查
            q_path = scale_only_path
            k_path = scale_transpose_path

            original_q = q_path["original_node"]
            original_k = k_path["original_node"]
            q_scaled = q_path["producer"] if q_path["is_scale_only"] else None
            k_scaled = None
            transpose_k = None

            # 确定k_scaled和transpose_k
            if k_path["producer"] and k_path["producer"].op_type in ["Mul", "Div"]:
                # Scale -> Transpose case
                k_scaled = k_path["producer"]
                # 找到transpose节点
                temp_producer = k_path["producer"]
                for inp in temp_producer.inputs:
                    if inp["name"] != k_path["input_name"] and inp["name"] in dag_model.initializers:
                        continue  # 这是scale值
                    else:
                        temp_input_name = inp["name"]
                        temp_input_producer = producer.get(temp_input_name)
                        if temp_input_producer and temp_input_producer.op_type == "Transpose":
                            transpose_k = temp_input_producer
                            break
            elif k_path["producer"] and k_path["producer"].op_type == "Transpose":
                # Transpose -> Scale case
                transpose_k = k_path["producer"]
                # 找到scale节点
                temp_input_name = k_path["producer"].inputs[0]["name"]
                temp_input_producer = producer.get(temp_input_name)
                if temp_input_producer and temp_input_producer.op_type in ["Mul", "Div"]:
                    k_scaled = temp_input_producer

            # Softmax的输出应该连接到一个MatMul（用于与V相乘）
            softmax_output_name = softmax_node.outputs[0]["name"]
            matmul_v_candidates = []
            
            # 使用consumer映射来查找使用softmax输出的MatMul节点
            for consumer_node, _ in consumers.get(softmax_output_name, []):
                if consumer_node.op_type == "MatMul":
                    matmul_v_candidates.append(consumer_node)

            if not matmul_v_candidates:
                continue

            # 4. 检查V是否直接连接到第二个MatMul
            matched = False
            for matmul_v_node in matmul_v_candidates:
                v_input_name = None
                for inp in matmul_v_node.inputs:
                    if inp["name"] != softmax_output_name:  # 找到V的输入
                        v_input_name = inp["name"]
                        break

                if not v_input_name:
                    continue

                original_v = producer.get(v_input_name)

                if original_v:
                    # 验证找到了完整的Attention模式
                    match = {
                        "q_node": original_q,
                        "k_node": original_k,
                        "v_node": original_v,
                        "q_scaled": q_scaled,
                        "k_scaled": k_scaled,
                        "transpose_k": transpose_k,
                        "matmul_qk": matmul_qk_node,
                        "add_node": add_node,
                        "softmax_node": softmax_node,
                        "matmul_v": matmul_v_node,
                        "att_mask": att_mask_input,
                    }

                    matches.append(match)
                    matched = True
                    break  # 找到一个匹配即可
        print(f"Found {len(matches)} attention patterns for potential fusion")
        return matches

    @staticmethod
    def fold_attention(dag_model, match) -> bool:
        """
        Fold the matched attention pattern into a single Attention operator.
        """
        q_node = match["q_node"]
        k_node = match["k_node"]
        v_node = match["v_node"]
        q_scaled = match["q_scaled"]
        k_scaled = match["k_scaled"]
        transpose_k = match["transpose_k"]
        matmul_qk = match["matmul_qk"]
        add_node = match["add_node"]
        softmax_node = match["softmax_node"]
        matmul_v = match["matmul_v"]
        att_mask = match["att_mask"]

        # 提取scale值
        scale_val = 1.0  # 默认值
        if q_scaled and q_scaled.op_type in ["Mul", "Div"]:
            for inp in q_scaled.inputs:
                if inp["name"] in dag_model.initializers:
                    scale_tensor = dag_model.initializers[inp["name"]]
                    scale_array = numpy_helper.to_array(scale_tensor)
                    if q_scaled.op_type == "Div":
                        scale_val = 1.0 / scale_array.item()  # 如果是除法，需要倒置
                    else:
                        scale_val = scale_array.item()  # 如果是乘法，直接使用
                    break

        # 构建新的Attention节点
        attention_inputs = [
            q_node.outputs[0],  # Q
            k_node.outputs[0],  # K
            v_node.outputs[0],  # V
        ]

        # 如果有attention mask，添加到输入
        if att_mask:
            attention_inputs.append(att_mask)

        # 设置Attention算子的属性
        attention_attrs = {}
        if scale_val != 1.0:
            attention_attrs["scale"] = scale_val

        # 确定unified_layout和其他属性（根据输入张量的形状推断）
        # 这里需要根据具体的Q/K/V形状来推断
        q_shape = q_node.outputs[0]["shape"]
        k_shape = k_node.outputs[0]["shape"]

        # 根据形状推断num_heads等参数
        # 通常形状为 [batch_size, seq_len, hidden_size] 或 [batch_size, num_heads, seq_len, head_dim]
        if len(q_shape) == 4:  # [batch, num_heads, seq_len, head_dim]
            num_heads = q_shape[1]
            attention_attrs["num_heads"] = num_heads

        attention_node = Node(
            op_type="Attention",
            name=f"Attention_fused_{q_node.name}_{k_node.name}_{v_node.name}",
            attributes=attention_attrs,
            inputs=attention_inputs,
            outputs=matmul_v.outputs[:],  # 使用原来的最终输出
        )

        # 删除旧的节点
        nodes_to_delete = [q_scaled, k_scaled, transpose_k, matmul_qk, softmax_node, matmul_v]
        if add_node:
            nodes_to_delete.append(add_node)

        for node in nodes_to_delete:
            if node and node.name in dag_model.nodes:
                del dag_model.nodes[node.name]

        # 添加新的Attention节点
        dag_model.nodes[attention_node.name] = attention_node

        print(f"Fused Attention pattern with Q:{q_node.name}, K:{k_node.name}, V:{v_node.name}")
        return True


class InitializerMerger:
    """Class for merging and manipulating initializers."""

    @staticmethod
    def merge_initializers(dag_model):
        """
        Merge batch normalization parameters into a single tensor [4, N]
        where N is the number of channels.

        Layout:
        Row 0: scale/weight (default: 1.0)
        Row 1: bias (default: 0.0)
        Row 2: mean (required)
        Row 3: variance (required)

        This modifies the dag_model by:
        1. Finding batch normalization nodes
        2. Merging their 4 input parameters into one
        3. Updating the nodes and initializers accordingly
        """

        # Find all batch normalization nodes
        bn_nodes = []
        for node_name, node in dag_model.nodes.items():
            if node.op_type == "BatchNormalization":
                bn_nodes.append((node_name, node))

        # Keep track of which initializers have been merged
        merged_initializers = set()

        for node_name, node in bn_nodes:
            # BatchNormalization typically has 5 inputs:
            # input, scale, bias, mean, variance
            if len(node.inputs) < 5:
                print(f"Warning: BatchNormalization node {node.name} has less than 5 inputs")
                continue

            # Get the names of the parameters
            # input[0] is the actual input data
            # inputs[1-4] are scale, bias, mean, variance respectively
            input_data = node.inputs[0]  # input data
            scale_name = node.inputs[1]["name"]  # scale/weight
            bias_name = node.inputs[2]["name"]  # bias
            mean_name = node.inputs[3]["name"]  # mean
            var_name = node.inputs[4]["name"]  # variance

            # Check if all required parameters exist
            required_params = [mean_name, var_name]
            for param_name in required_params:
                if param_name not in dag_model.initializers:
                    print(f"Error: Required parameter {param_name} not found in initializers")
                    continue

            # Get the parameter arrays
            try:
                mean_array = numpy_helper.to_array(dag_model.initializers[mean_name])
                var_array = numpy_helper.to_array(dag_model.initializers[var_name])

                # Validate that mean and variance have the same shape
                if mean_array.shape != var_array.shape:
                    print(
                        f"Error: Mean {mean_array.shape} and variance {var_array.shape} have different shapes"
                    )
                    continue

                # Handle optional parameters with defaults
                if scale_name in dag_model.initializers:
                    scale_array = numpy_helper.to_array(dag_model.initializers[scale_name])
                else:
                    scale_array = np.ones_like(mean_array, dtype=mean_array.dtype)

                if bias_name in dag_model.initializers:
                    bias_array = numpy_helper.to_array(dag_model.initializers[bias_name])
                else:
                    bias_array = np.zeros_like(mean_array, dtype=mean_array.dtype)

                # where N is the number of elements in each parameter
                N = mean_array.size
                padded_N = ((N + 3) // 4) * 4
                merged_data = np.zeros((4 * padded_N), dtype=mean_array.dtype)

                # Reshape all arrays to 1D for consistent indexing
                scale_flat = scale_array.flatten()
                bias_flat = bias_array.flatten()
                mean_flat = mean_array.flatten()
                var_flat = var_array.flatten()
                if scale_flat.size < padded_N:
                    scale_flat = np.pad(
                        scale_flat, (0, padded_N - scale_flat.size), constant_values=1.0
                    )
                    bias_flat = np.pad(
                        bias_flat, (0, padded_N - bias_flat.size), constant_values=0.0
                    )
                    mean_flat = np.pad(
                        mean_flat, (0, padded_N - mean_flat.size), constant_values=0.0
                    )
                    var_flat = np.pad(var_flat, (0, padded_N - var_flat.size), constant_values=1.0)

                # Fill the merged tensor
                for i in range(padded_N // 4):
                    base_idx = i * 4
                    # Reorganize data to match C++ implementation (interleaved format)
                    # Each group of 16 elements contains 4 vec4: scale, bias, mean, variance
                    for j in range(4):
                        if base_idx + j < N:
                            merged_data[i * 16 + j] = scale_flat[base_idx + j]  # scale
                            merged_data[i * 16 + 4 + j] = bias_flat[base_idx + j]  # bias
                            merged_data[i * 16 + 8 + j] = mean_flat[base_idx + j]  # mean
                            merged_data[i * 16 + 12 + j] = var_flat[base_idx + j]  # variance
                        else:
                            merged_data[i * 16 + j] = 1.0  # scale default
                            merged_data[i * 16 + 4 + j] = 0.0  # bias default
                            merged_data[i * 16 + 8 + j] = 0.0  # mean default
                            merged_data[i * 16 + 12 + j] = 1.0  # variance default
                # Create a new initializer name
                merged_name = f"{node.name}_bn_params"

                # Convert back to ONNX tensor
                merged_tensor = numpy_helper.from_array(merged_data, merged_name)

                # Add the merged tensor to initializers
                dag_model.initializers[merged_name] = merged_tensor

                # Update the node to use the merged parameter
                # Change inputs from 5 to 2: [input, merged_params]
                new_inputs = [
                    input_data,  # Original input data (index 0)
                    {"name": merged_name, "shape": [4 * padded_N]},  # Merged parameters
                ]

                node.inputs = new_inputs

                # Mark original initializers for removal
                merged_initializers.update([scale_name, bias_name, mean_name, var_name])

                print(
                    f"Merged batchnorm for {node.name}: "
                    f"scale({scale_name}), bias({bias_name}), mean({mean_name}), var({var_name}) "
                    f"-> merged({merged_name})"
                )

            except Exception as e:
                print(f"Error processing BatchNormalization node {node.name}: {e}")
                continue

        # Remove the original individual initializers
        for initializer_name in merged_initializers:
            if initializer_name in dag_model.initializers:
                del dag_model.initializers[initializer_name]

        print(
            f"Merged {len(bn_nodes)} BatchNormalization nodes, removed {len(merged_initializers)} initializers"
        )

    @staticmethod
    def convert_flat_to_reshape(dag_model):
        """
        Convert Flat nodes to Reshape nodes with explicit shapes.

        Flatten operation flattens the input tensor into a 2D tensor, keeping dimensions
        up to axis-1 and flattening the rest into the second dimension.
        """
        nodes_to_update = []

        for node_name, node in dag_model.nodes.items():
            if node.op_type == "Flatten":
                # Get input shape
                if len(node.inputs) > 0 and len(node.inputs[0]["shape"]) > 0:
                    input_shape = node.inputs[0]["shape"]
                    print(f"Flatten node {node.name} input shape: {input_shape}")

                    # Get axis attribute (default is 1 according to ONNX spec)
                    axis = node.attributes.get("axis", 1)

                    # Calculate output shape for flatten:
                    # First part: product of dimensions from 0 to axis-1
                    # Second part: product of dimensions from axis to end
                    if axis == 0:
                        first_part = 1
                    else:
                        first_part = 1
                        for i in range(axis):
                            first_part *= input_shape[i]

                    second_part = 1
                    for i in range(axis, len(input_shape)):
                        second_part *= input_shape[i]

                    output_shape = [first_part, second_part]

                    # Create new reshape node
                    reshape_node = Node(
                        op_type="Reshape",
                        name=node.name,
                        attributes={},
                        inputs=node.inputs[:],  # Copy original inputs
                        outputs=node.outputs[:],  # Copy original outputs
                    )

                    # Add shape tensor as second input
                    shape_tensor_name = node.name + "_shape"
                    shape_tensor = np.array(output_shape, dtype=np.int64)
                    shape_initializer = numpy_helper.from_array(shape_tensor, shape_tensor_name)
                    dag_model.initializers[shape_tensor_name] = shape_initializer

                    # Add the shape tensor as the second input to reshape
                    reshape_node.inputs.append(
                        {"name": shape_tensor_name, "shape": list(shape_tensor.shape)}
                    )

                    nodes_to_update.append((node_name, reshape_node))
                    print(
                        f"Converted Flatten node '{node.name}' to Reshape with shape {output_shape} (axis={axis})"
                    )
                else:
                    # If we can't determine the shape, keep the original node
                    print(
                        f"Warning: Could not convert Flatten node '{node.name}' - missing shape info"
                    )

        # Apply updates
        for old_name, new_node in nodes_to_update:
            del dag_model.nodes[old_name]
            dag_model.nodes[new_node.name] = new_node

    @staticmethod
    def remove_redundant_reshape(dag_model):
        """
        Remove redundant reshape nodes where input and output shapes are the same.
        Updates connections so that the reshape's input becomes the next node's input.
        Also cleans up unused initializers.
        """
        # Build mapping of output names to producing nodes
        producer = {}
        for node in dag_model.nodes.values():
            for out in node.outputs:
                producer[out["name"]] = node

        # Track which nodes to remove
        to_remove = []
        # Map from reshape output names to their input names
        reshape_remap = {}
        # Track initializers used by redundant reshapes
        redundant_initializer_names = set()

        # First pass: identify redundant reshapes and build remapping
        for node_name, node in dag_model.nodes.items():
            if node.op_type == "Reshape":
                # Check if input and output shapes are the same
                if (
                    len(node.inputs) >= 1
                    and len(node.outputs) >= 1
                    and node.inputs[0]["shape"] == node.outputs[0]["shape"]
                ):

                    # This is a redundant reshape node
                    input_name = node.inputs[0]["name"]
                    output_name = node.outputs[0]["name"]

                    # Record the mapping for remapping
                    reshape_remap[output_name] = input_name
                    # Mark this reshape node for removal
                    to_remove.append(node_name)

                    # Collect initializers used by this reshape node (typically the shape tensor)
                    for inp in node.inputs[
                        1:
                    ]:  # Skip the first input (data), consider the shape input
                        if inp["name"] in dag_model.initializers:
                            redundant_initializer_names.add(inp["name"])

                    print(f"Identified redundant Reshape node: {node.name}")

        # Check if the collected initializers are used by any other nodes
        # If not, they should be removed
        initializers_to_remove = set()
        if redundant_initializer_names:
            # Build a set of all tensor names used by all nodes (except the ones we're removing)
            used_tensors = set()
            for node_name, node in dag_model.nodes.items():
                if node_name not in to_remove:  # Skip nodes we're going to remove
                    for inp in node.inputs:
                        used_tensors.add(inp["name"])
                    for out in node.outputs:
                        used_tensors.add(out["name"])

            # Check if any of our redundant initializers are actually used elsewhere
            for initializer_name in redundant_initializer_names:
                if initializer_name not in used_tensors:
                    initializers_to_remove.add(initializer_name)

        # Second pass: update all nodes that reference the removed reshape outputs
        for node in dag_model.nodes.values():
            if node.name in to_remove:
                continue  # Skip the nodes we're removing

            # Update inputs that reference removed reshape outputs
            for inp in node.inputs:
                if inp["name"] in reshape_remap:
                    old_name = inp["name"]
                    inp["name"] = reshape_remap[old_name]
                    # Also update the shape if needed (should be the same)
                    # Find the source node/input to get the correct shape
                    print(f"Remapped input {old_name} to {inp['name']} in node {node.name}")

        # Remove the marked nodes
        if to_remove:
            for node_name in to_remove:
                if node_name in dag_model.nodes:
                    del dag_model.nodes[node_name]
            print(f"Removed {len(to_remove)} redundant reshape nodes")

        # Remove unused initializers
        if initializers_to_remove:
            for initializer_name in initializers_to_remove:
                if initializer_name in dag_model.initializers:
                    del dag_model.initializers[initializer_name]
            print(
                f"Removed {len(initializers_to_remove)} unused initializers: {initializers_to_remove}"
            )

    @staticmethod
    def move_input_tensor_to_attr(dag_model):
        """
        将一些算子input中包含的仅rank长度的tensor转换为attribute。
        比如resize算子中的scales、sizes, pad算子中的pads等, 这些都是小型一维张量，
        对于这类tensor可以直接转为attribute, 同时将node inputs中对应tensor置为空,
        对应的initializer也删除.
        """
        initializers_to_remove = set()

        SPECIAL_OPS = {
            "Resize": [(2, "scales"), (3, "sizes")],  # inputs[2]=scales, inputs[3]=sizes
            "Pad": [(1, "pads")],  # inputs[1]=pads
            # 'Slice': [(1, 'starts'), (2, 'ends'), (3, 'axes'), (4, 'steps')],  # 多个参数
        }

        for node in dag_model.nodes.values():
            op_type = node.op_type
            if op_type not in SPECIAL_OPS:
                continue

            target_inputs = SPECIAL_OPS[op_type]

            for idx, attr_name in target_inputs:
                if idx >= len(node.inputs):
                    continue

                input_tensor = node.inputs[idx]
                tensor_name = input_tensor["name"]
                print("Checking input tensor: ", tensor_name)

                if not tensor_name or tensor_name not in dag_model.initializers:
                    continue

                initializer = dag_model.initializers[tensor_name]

                # 检查是否为一维数组且长度较短（通常是rank长度，一般不超过8）
                if len(initializer.dims) == 1 and 0 < initializer.dims[0] <= 8:
                    tensor_data = numpy_helper.to_array(initializer)

                    if not hasattr(node, "attributes") or node.attributes is None:
                        node.attributes = {}

                    if tensor_data.dtype in [np.float32, np.float64]:
                        node.attributes[attr_name] = tensor_data.tolist()
                    elif tensor_data.dtype in [np.int32, np.int64]:
                        node.attributes[attr_name] = tensor_data.tolist()
                    else:
                        node.attributes[attr_name] = tensor_data.tolist()

                    del node.inputs[idx]

                    initializers_to_remove.add(tensor_name)
            print(node)

        for initializer_name in initializers_to_remove:
            if initializer_name in dag_model.initializers:
                del dag_model.initializers[initializer_name]

        print(f"Converted {len(initializers_to_remove)} tensor inputs to attributes")


class Quantizer:
    """Class for quantizing model weights."""

    @staticmethod
    def quantize_to_fp16_selective(dag_model):
        """
        Selectively quantize model weights to FP16 based on operator type and parameter sensitivity.

        This function converts only appropriate FP32 initializers to FP16, considering:
        1. Which operator uses the initializer
        2. What role the initializer plays (weights vs. batch norm parameters)
        3. Sensitivity of different parameter types to quantization

        Generally safe to convert:
        - Convolution weights
        - Gemm/Linear weights
        - Recurrent weights

        Usually NOT safe to convert:
        - BatchNorm parameters (scale, bias, mean, var)
        - Small embedding tables
        """
        print("Selectively quantizing model to FP16...")

        converted_count = 0
        skipped_count = 0

        # Build mapping from initializer names to their consumers
        initializer_consumers = defaultdict(list)
        for node in dag_model.nodes.values():
            for inp in node.inputs:
                initializer_consumers[inp["name"]].append(node)

        # Data types that should remain unchanged
        preserve_types = {
            onnx.TensorProto.UINT8,
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT16,
            onnx.TensorProto.INT16,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.UINT32,
            onnx.TensorProto.UINT64,
            onnx.TensorProto.BOOL,
        }

        # Operators whose weights are usually safe to quantize
        safe_weight_operators = {"Conv", "Gemm", "MatMul", "ConvTranspose", "LSTM", "GRU", "RNN"}

        # Parameters that are usually sensitive to FP16 quantization
        sensitive_parameters = {"BatchNormalization"}

        for name, initializer in dag_model.initializers.items():
            # Skip non-FP32 tensors
            if initializer.data_type != onnx.TensorProto.FLOAT:
                if initializer.data_type in preserve_types:
                    print(
                        f"Preserving {onnx.TensorProto.DataType.Name(initializer.data_type)} tensor '{name}'"
                    )
                elif initializer.data_type == onnx.TensorProto.FLOAT16:
                    print(f"Skipping already FP16 tensor '{name}'")
                else:
                    data_type_name = (
                        onnx.TensorProto.DataType.Name(initializer.data_type)
                        if initializer.data_type <= 16
                        else "UNKNOWN"
                    )
                    print(
                        f"Preserving {data_type_name} tensor '{name}' (type: {initializer.data_type})"
                    )
                skipped_count += 1
                continue

            # Check who consumes this initializer
            consumers = initializer_consumers.get(name, [])
            consumer_ops = {node.op_type for node in consumers}

            # Determine if this initializer should be quantized
            should_quantize = False
            reason = ""

            arr = numpy_helper.to_array(initializer)
            if any(op in safe_weight_operators for op in consumer_ops):
                # This initializer is consumed by operators known to be safe for FP16
                should_quantize = True
                reason = f"consumed by safe operators {consumer_ops}"
            elif not consumers:
                # Orphaned initializer - better to preserve
                should_quantize = False
                reason = "no consumers"
            elif any(op in sensitive_parameters for op in consumer_ops):
                # Consumed by sensitive operators like BatchNormalization
                should_quantize = False
                reason = f"consumed by sensitive operators {consumer_ops}"
            else:
                # Default behavior - check size (small tensors might be sensitive)
                should_quantize = False
                reason = f"small tensor ({arr.size} elements)"

            if should_quantize:
                # Convert to numpy array
                arr = numpy_helper.to_array(initializer)

                # Convert to FP16
                arr_fp16 = arr.astype(np.float16)

                # Convert back to ONNX tensor with FP16 data type
                fp16_initializer = numpy_helper.from_array(arr_fp16, name)

                # Update the initializer in the model
                dag_model.initializers[name] = fp16_initializer
                print(f"Converted FP32 tensor '{name}' to FP16 ({reason})")
                print(f"New shape: {fp16_initializer.dims}")
                converted_count += 1
            else:
                skipped_count += 1

        print(f"Converted {converted_count} FP32 tensors to FP16")
        print(f"Preserved {skipped_count} tensors")

    @staticmethod
    def quantize_to_int8_weight_only(dag_model):
        """
        Quantize model weights to INT8 using weight-only quantization.
        This function converts appropriate FP32 initializers to INT8, preserving scale information.

        For each tensor that is quantized:
        - The original FP32 weights are quantized to INT8
        - A scale tensor is created and stored as a separate initializer
        - The scale is calculated per-channel or per-tensor depending on tensor size

        Generally safe to convert:
        - Convolution weights
        - Gemm/Linear weights
        - Recurrent weights
        """
        print("Applying weight-only INT8 quantization...")

        converted_count = 0
        skipped_count = 0

        # Build mapping from initializer names to their consumers
        initializer_consumers = defaultdict(list)
        for node in dag_model.nodes.values():
            for inp in node.inputs:
                initializer_consumers[inp["name"]].append(node)

        # Data types that should remain unchanged
        preserve_types = {
            onnx.TensorProto.UINT8,
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT16,
            onnx.TensorProto.INT16,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.UINT32,
            onnx.TensorProto.UINT64,
            onnx.TensorProto.BOOL,
        }

        # Operators whose weights are usually safe to quantize to INT8
        safe_weight_operators = {"Conv", "MatMul", "ConvTranspose", "LSTM", "GRU", "RNN"}

        # Parameters that are usually sensitive to INT8 quantization
        sensitive_parameters = {"BatchNormalization", "LayerNormalization", "GroupNormalization"}

        # Get a list of keys to iterate over, to avoid modifying the dict during iteration
        initializers_keys = list(dag_model.initializers.keys())

        for name in initializers_keys:
            initializer = dag_model.initializers[name]

            # Skip non-FP32 tensors
            if initializer.data_type != onnx.TensorProto.FLOAT:
                if initializer.data_type in preserve_types:
                    print(
                        f"Preserving {onnx.TensorProto.DataType.Name(initializer.data_type)} tensor '{name}'"
                    )
                elif initializer.data_type == onnx.TensorProto.FLOAT16:
                    print(f"Skipping already FP16 tensor '{name}'")
                else:
                    data_type_name = (
                        onnx.TensorProto.DataType.Name(initializer.data_type)
                        if initializer.data_type <= 16
                        else "UNKNOWN"
                    )
                    print(
                        f"Preserving {data_type_name} tensor '{name}' (type: {initializer.data_type})"
                    )
                skipped_count += 1
                continue

            # Check who consumes this initializer
            consumers = initializer_consumers.get(name, [])
            consumer_ops = {node.op_type for node in consumers}

            # Determine if this initializer should be quantized
            should_quantize = False
            reason = ""

            if any(op in safe_weight_operators for op in consumer_ops):
                # This initializer is consumed by operators known to be safe for INT8 quantization
                should_quantize = True
                reason = f"consumed by safe operators {consumer_ops}"
            elif not consumers:
                # Orphaned initializer - better to preserve
                should_quantize = False
                reason = "no consumers"
            elif any(op in sensitive_parameters for op in consumer_ops):
                # Consumed by sensitive operators like BatchNormalization
                should_quantize = False
                reason = f"consumed by sensitive operators {consumer_ops}"
            else:
                # Default behavior - check size (small tensors might be sensitive)
                should_quantize = False
            # Additional check: skip bias tensors
            # Bias tensors are typically 1D and have small sizes
            arr = numpy_helper.to_array(initializer)
            if len(arr.shape) == 1:  # Common bias tensor characteristics
                # Check if it's connected to a bias input in operations like Conv, Gemm, etc.
                is_bias = False
                for node in consumers:
                    op_type = node.op_type
                    if op_type in ["Conv", "ConvTranspose"]:
                        # Check if this tensor is the bias input (usually the 3rd input for Conv, 2nd for Gemm)
                        for idx, inp in enumerate(node.inputs):
                            if inp["name"] == name:
                                # For Conv: 0=inputs, 1=weights, 2=bias
                                if (op_type == "Conv" and idx == 2) or (
                                    op_type == "ConvTranspose" and idx == 2
                                ):
                                    is_bias = True
                                    break
                if is_bias:
                    print(f"Preserving bias tensor '{name}' as FP32")
                    skipped_count += 1
                    should_quantize = False
            if should_quantize:
                # Convert to numpy array
                arr = numpy_helper.to_array(initializer)
                original_fp32 = arr.copy()

                # Determine quantization axis based on operator type and tensor shape
                axis = None
                if len(arr.shape) >= 2:
                    # For multi-dimensional weights, use operator-specific quantization axis
                    for node in consumers:
                        op_type = node.op_type
                        if op_type == "Conv":
                            # Conv weights: [C_out, C_in, K, K] - quantize per output channel
                            # Reduce along (C_in, K, K) dimensions -> axis=(1, 2, 3)
                            if len(arr.shape) == 4:
                                axis = (1, 2, 3)
                            elif len(arr.shape) == 3:
                                axis = (1, 2)
                            else:
                                axis = 0
                            break
                        elif op_type == "ConvTranspose":
                            # ConvTranspose weights: [C_in, C_out, K, K] - quantize per output channel
                            # Reduce along (C_in, K, K) dimensions -> axis=(0, 2, 3)
                            if len(arr.shape) == 4:
                                axis = (0, 2, 3)
                            else:
                                # For other shapes, default to axis=1
                                axis = 1
                            break
                        elif op_type == "Gemm":
                            # Gemm weights: [out, in] - quantize per output dimension
                            # Reduce along in dimension -> axis=1
                            if len(arr.shape) == 2:
                                axis = 1
                            else:
                                # For other shapes, default to axis=0
                                axis = 0
                            break
                        elif op_type == "MatMul":
                            # MatMul weights: typically [in, out] - quantize per output dimension
                            # Reduce along in dimension -> axis=0
                            if len(arr.shape) == 2:
                                axis = 0
                            else:
                                # For other shapes, default to axis=1
                                axis = 1
                            break
                        elif op_type in ["LSTM", "GRU"]:
                            # LSTM/GRU weights: [D, 4H, I] or [D, 4H, H] - quantize per output dimension
                            # Reduce along I or H dimension -> axis=2
                            if len(arr.shape) == 3:
                                axis = 2
                            else:
                                # For other shapes, default to axis=0
                                axis = 0
                            break
                        else:
                            # Default to axis=0 for other operators
                            axis = 0
                else:
                    # For 1D or smaller tensors, use per-tensor quantization
                    axis = None

                # Perform INT8 quantization
                if axis is not None:
                    # Calculate scale per specified axis
                    amax = np.amax(np.abs(arr), axis=axis, keepdims=True)
                    scale_keepdims = amax / 127.0  # INT8 range is [-128, 127]

                    # Avoid division by zero
                    scale_keepdims = np.where(scale_keepdims == 0, 1.0, scale_keepdims)

                    arr_int8 = np.round(arr / scale_keepdims).astype(np.int8)

                    # For most operators (except LSTM/GRU), create 1D scale for storage
                    if not any(op in ["LSTM", "GRU"] for op in consumer_ops):
                        # Calculate scale per axis but flatten to 1D for storage
                        amax_1d = np.amax(np.abs(arr), axis=axis, keepdims=False)  # Flatten scale
                        scale = amax_1d / 127.0

                        # Avoid division by zero
                        scale = np.where(scale == 0, 1.0, scale)

                        # For different operator types, ensure correct shape
                        if "Conv" in consumer_ops and len(arr.shape) == 4:
                            # Conv: [C_out, C_in, K, K], axis=(1,2,3) -> scale should be [C_out]
                            scale = scale.reshape(arr.shape[0])
                        elif "ConvTranspose" in consumer_ops and len(arr.shape) == 4:
                            # ConvTranspose: [C_in, C_out, K, K], axis=(0,2,3) -> scale should be [C_out]
                            scale = scale.reshape(arr.shape[1])
                        elif "Gemm" in consumer_ops and len(arr.shape) == 2:
                            # Gemm: [out, in], axis=1 -> scale should be [out]
                            scale = scale.reshape(arr.shape[0])
                        elif "MatMul" in consumer_ops and len(arr.shape) == 2:
                            # MatMul: [in, out], axis=0 -> scale should be [out]
                            scale = scale.reshape(arr.shape[1])
                        else:
                            # For other cases, flatten to 1D
                            scale = scale.flatten()
                    else:
                        # For LSTM/GRU, keep the reduced scale but not the original shape
                        scale = amax_1d / 127.0
                        scale = np.where(scale == 0, 1.0, scale)
                else:
                    # For 1D or smaller tensors, use per-tensor quantization
                    amax = np.amax(np.abs(arr))
                    scale_keepdims = amax / 127.0 if amax != 0 else 1.0
                    scale = scale_keepdims  # For per-tensor, scale is scalar
                    arr_int8 = np.round(arr / scale_keepdims).astype(np.int8)

                # === 反量化: INT8 -> FP32 ===
                if axis is not None:
                    # Broadcast scale back to original shape for dequantization
                    scale_broadcast = (
                        np.expand_dims(scale, axis=axis) if np.ndim(scale) > 0 else scale
                    )
                    dequantized = arr_int8.astype(np.float32) * scale_broadcast
                else:
                    dequantized = arr_int8.astype(np.float32) * scale
                # === 计算误差指标 ===
                diff = dequantized - original_fp32
                mse = np.mean(diff**2)
                mae = np.mean(np.abs(diff))
                max_abs_error = np.max(np.abs(diff))
                # Relative error (avoid division by zero)
                rel_error = np.abs(diff) / (np.abs(original_fp32) + 1e-8)
                mean_rel_error = np.mean(rel_error)
                max_rel_error = np.max(rel_error)
                print(f"Quantized '{name}':")
                print(f"  Shape: {original_fp32.shape}")
                print(f"  Scale shape: {scale.shape if hasattr(scale, 'shape') else 'scalar'}")
                print(f"  MSE: {mse:.6e}, MAE: {mae:.6e}")
                print(f"  Max Abs Error: {max_abs_error:.6e}")
                print(f"  Mean Rel Error: {mean_rel_error:.2%}, Max Rel Error: {max_rel_error:.2%}")

                # Create INT8 initializer for quantized weights
                int8_initializer = numpy_helper.from_array(arr_int8.astype(np.int8), name)
                int8_initializer.data_type = onnx.TensorProto.INT8

                # Create scale initializer
                scale_name = f"{name}_scale"
                scale_initializer = numpy_helper.from_array(scale, scale_name)
                scale_initializer.data_type = onnx.TensorProto.FLOAT

                # Update the model: replace original with INT8 weights and add scale
                dag_model.initializers[name] = int8_initializer

                scale_name = f"{name}_scale"
                scale_initializer = numpy_helper.from_array(scale, scale_name)
                scale_initializer.data_type = onnx.TensorProto.FLOAT
                dag_model.initializers[scale_name] = scale_initializer

                # Add scale as input to the nodes that consume the original initializer
                for node in consumers:
                    # Add scale tensor as an additional input
                    scale_input = {
                        "name": scale_name,
                        "shape": list(scale.shape) if hasattr(scale, "shape") else [],
                    }
                    node.inputs.append(scale_input)

                print(
                    f"Converted FP32 tensor '{name}' to INT8 with scale tensor '{scale_name}' ({reason})"
                )
                print(f"Original shape: {initializer.dims}, scale shape: {scale_initializer.dims}")
                converted_count += 1
            else:
                skipped_count += 1

        print(f"Converted {converted_count} FP32 tensors to INT8 with scale information")
        print(f"Preserved {skipped_count} tensors")
        print(f"Total initializers after quantization: {len(dag_model.initializers)}")
