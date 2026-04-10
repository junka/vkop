"""Main model converter implementation."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

try:
    from .dag import DAGBasedModel, Node
    from .optimizer import FusionOptimizer, InitializerMerger, ONNXOptimizer, Quantizer
except ImportError:
    from dag import DAGBasedModel, Node
    from optimizer import FusionOptimizer, InitializerMerger, ONNXOptimizer, Quantizer


class ModelConverter:
    """Main class for converting ONNX models to DAG-based format."""

    def __init__(self):
        self.optimizer = ONNXOptimizer()
        self.fusion_optimizer = FusionOptimizer()
        self.initializer_merger = InitializerMerger()
        self.quantizer = Quantizer()

    def parse_onnx_model(self, onnx_path: str, batch_size: int = 1) -> DAGBasedModel:
        """Parse and convert ONNX model to DAG format."""
        model = onnx.load(onnx_path)

        print("Optimizing ONNX model...")
        model = self.optimizer.optimize_model(model, batch_size)

        onnx.save(model, "optimized_" + Path(onnx_path).name)

        dag_model = DAGBasedModel()

        graph = model.graph
        try:
            onnx.checker.check_model(model, full_check=True)
            print("ONNX model passed full validation.")
        except onnx.checker.ValidationError as e:
            print(f"ONNX model full validation failed: {e}")

        if not self.optimizer.is_topologically_sortable(graph):
            print("Graph is not topologically sorted. Please sort it before proceeding.")
            return None

        # Add initializers
        for initializer in graph.initializer:
            dag_model.initializers[initializer.name] = initializer

        initializer_names = {init.name for init in graph.initializer}

        # Add inputs
        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            tensor_type = inp.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") else 1 for dim in tensor_type.shape.dim
            ]
            print("Graph input:", inp.name, "of shape:", shape_dims)
            dag_model.inputs.append({"name": inp.name, "shape": shape_dims})

        # Add outputs
        for out in graph.output:
            tensor_type = out.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") else 1 for dim in tensor_type.shape.dim
            ]
            print("Graph output:", out.name, "of shape:", shape_dims)
            dag_model.outputs.append({"name": out.name, "shape": shape_dims})

        modified_shapes = defaultdict(list)
        ELEMWISE_OPS = {
            "Add",
            "And",
            "Div",
            "Equal",
            "Greater",
            "Less",
            "Max",
            "Mean",
            "Min",
            "Mul",
            "Or",
            "Pow",
            "Sub",
            "Sum",
            "Where",
            "Xor",
        }
        for i, node in enumerate(graph.node):
            if not node.name:
                # 使用 op_type + index 生成唯一名，避免冲突
                node.name = f"{node.op_type}_{i}"
        for node in graph.node:
            print("Processing node:", node.name, "of type:", node.op_type)
            attributes = {}
            for attr in node.attribute:
                # https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
                # message AttributeProto
                # enum AttributeType {
                #     UNDEFINED = 0;
                #     FLOAT = 1;
                #     INT = 2;
                #     STRING = 3;
                #     TENSOR = 4;
                #     GRAPH = 5;
                #     SPARSE_TENSOR = 11;
                #     TYPE_PROTO = 13;

                #     FLOATS = 6;
                #     INTS = 7;
                #     STRINGS = 8;
                #     TENSORS = 9;
                #     GRAPHS = 10;
                #     SPARSE_TENSORS = 12;
                #     TYPE_PROTOS = 14;
                # }

                if attr.type == onnx.AttributeProto.INT:
                    attributes[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attributes[attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    attributes[attr.name] = attr.s.decode("utf-8")
                elif attr.type == onnx.AttributeProto.TENSOR:
                    attributes[attr.name] = numpy_helper.to_array(attr.t)
                elif attr.type == onnx.AttributeProto.INTS:
                    attributes[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attributes[attr.name] = list(attr.floats)
                else:
                    print(
                        f"Warning: Unsupported attribute type {attr.type} for attribute {attr.name}"
                    )

            outputs_with_shape = []
            for output_name in node.output:
                print("  Output:", output_name)
                # Find the output tensor to get its shape
                output_tensor = None
                for value_info in graph.value_info:
                    if value_info.name == output_name:
                        output_tensor = value_info
                        break
                if output_tensor is None:
                    for out in graph.output:
                        if out.name == output_name:
                            output_tensor = out
                            break
                if output_tensor is None:
                    print(f"Warning: Output tensor {output_name} not found in graph.")
                    outputs_with_shape.append({"name": output_name, "shape": []})
                    continue
                tensor_type = output_tensor.type.tensor_type
                shape_dims = [
                    dim.dim_value if dim.HasField("dim_value") else 1
                    for dim in tensor_type.shape.dim
                ]
                #  GlobalAveragePool compression to 2d, so we can use storage instead of image
                if node.op_type == "GlobalAveragePool" and len(shape_dims) >= 2:
                    original_shape = shape_dims[:]
                    shape_dims = [original_shape[0], original_shape[1]]
                    print(f"Modified GlobalAveragePool shape from {original_shape} to {shape_dims}")
                    modified_shapes[output_name] = shape_dims

                outputs_with_shape.append({"name": output_name, "shape": shape_dims})

            inputs_with_shape = []
            for input_name in node.input:
                print("  Input:", input_name)
                # Find the input tensor to get its shape
                input_tensor = None
                for value_info in graph.value_info:
                    if value_info.name == input_name:
                        input_tensor = value_info
                        break
                if input_tensor is None:
                    for inp in graph.input:
                        if inp.name == input_name:
                            input_tensor = inp
                            break
                if input_tensor is None:
                    for out in graph.output:
                        if out.name == input_name:
                            input_tensor = out
                            break
                if input_tensor is None:
                    for initializer in graph.initializer:
                        if initializer.name == input_name:
                            data_type_map = {
                                1: onnx.TensorProto.FLOAT,
                                2: onnx.TensorProto.UINT8,
                                3: onnx.TensorProto.INT8,
                                4: onnx.TensorProto.UINT16,
                                5: onnx.TensorProto.INT16,
                                6: onnx.TensorProto.INT32,
                                7: onnx.TensorProto.INT64,
                                8: onnx.TensorProto.STRING,
                                9: onnx.TensorProto.BOOL,
                                10: onnx.TensorProto.FLOAT16,
                                11: onnx.TensorProto.DOUBLE,
                                12: onnx.TensorProto.UINT32,
                                13: onnx.TensorProto.UINT64,
                                14: onnx.TensorProto.COMPLEX64,
                                15: onnx.TensorProto.COMPLEX128,
                                16: onnx.TensorProto.BFLOAT16,
                            }
                            data_type = data_type_map.get(
                                initializer.data_type, onnx.TensorProto.UNDEFINED
                            )
                            input_tensor = onnx.helper.make_tensor_value_info(
                                initializer.name, data_type, initializer.dims
                            )
                            break

                if input_tensor is None:
                    print(f'Warning: Input tensor "{input_name}" not found in graph.')
                    inputs_with_shape.append({"name": input_name, "shape": []})
                    continue
                tensor_type = input_tensor.type.tensor_type
                shape_dims = [
                    dim.dim_value if dim.HasField("dim_value") else 1
                    for dim in tensor_type.shape.dim
                ]
                if input_name in modified_shapes and len(modified_shapes[input_name]) > 0:
                    shape_dims = modified_shapes[input_name]
                    print(f"Modified shape of input {input_name} to {shape_dims}")
                inputs_with_shape.append({"name": input_name, "shape": shape_dims})

            # if node.op_type in ELEMWISE_OPS and len(node.input) == 2:
            #     for i in range(len(inputs_with_shape)):
            #         if inputs_with_shape[i]["name"] == node.input[0]:
            #             id0 = i
            #         if inputs_with_shape[i]["name"] == node.input[1]:
            #             id1 = i
            #     if len(inputs_with_shape[id0]["shape"]) != len(inputs_with_shape[id1]["shape"]):
            #         shape = self.broadcast_shapes(
            #             inputs_with_shape[id0]["shape"], inputs_with_shape[id1]["shape"]
            #         )
            #         print(
            #             "  Elemwise operation with multiple inputs. Broadcasting shapes...", shape
            #         )
            #         if len(inputs_with_shape[id0]["shape"]) != len(shape):
            #             inputs_with_shape[id0]["shape"] = shape
            #             for initializer in graph.initializer:
            #                 if initializer.name == inputs_with_shape[id0]["name"]:
            #                     orig_init = dag_model.initializers[initializer.name]
            #                     orig_array = numpy_helper.to_array(orig_init)
            #                     if orig_array.size == 1:
            #                         scalar_val = orig_array.item()
            #                         expanded_data = np.full(
            #                             shape, scalar_val, dtype=orig_array.dtype
            #                         )
            #                     else:
            #                         expanded_data = np.broadcast_to(orig_array, shape).copy()
            #                     new_init = numpy_helper.from_array(
            #                         expanded_data, name=initializer.name
            #                     )
            #                     dag_model.initializers[initializer.name] = new_init
            #         if len(inputs_with_shape[id1]["shape"]) != len(shape):
            #             inputs_with_shape[id1]["shape"] = shape
            #             for initializer in graph.initializer:
            #                 if initializer.name == inputs_with_shape[id1]["name"]:
            #                     orig_init = dag_model.initializers[initializer.name]
            #                     orig_array = numpy_helper.to_array(orig_init)
            #                     if orig_array.size == 1:
            #                         scalar_val = orig_array.item()
            #                         expanded_data = np.full(
            #                             shape, scalar_val, dtype=orig_array.dtype
            #                         )
            #                     else:
            #                         expanded_data = np.broadcast_to(orig_array, shape).copy()
            #                     new_init = numpy_helper.from_array(
            #                         expanded_data, name=initializer.name
            #                     )
            #                     dag_model.initializers[initializer.name] = new_init

            new_node = Node(
                op_type=node.op_type,
                name=node.name,
                attributes=attributes,
                inputs=inputs_with_shape,
                outputs=outputs_with_shape,
            )
            dag_model.add_node(new_node)

        dag_model.build_dependencies()
        return dag_model

    @staticmethod
    def broadcast_shapes(shape_a, shape_b):
        """Compute broadcasted shape per ONNX rules."""
        if not shape_a and not shape_b:
            return []
        if not shape_a:
            return shape_b
        if not shape_b:
            return shape_a

        # Make copies and align from right
        a, b = list(shape_a), list(shape_b)
        len_a, len_b = len(a), len(b)
        max_len = max(len_a, len_b)

        # Left-pad with 1s
        a = [1] * (max_len - len_a) + a
        b = [1] * (max_len - len_b) + b

        result = []
        for da, db in zip(a, b):
            if da == db:
                result.append(da)
            elif da == 1:
                result.append(db)
            elif db == 1:
                result.append(da)
            else:
                raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
        return result

    def apply_optimizations(self, dag_model: DAGBasedModel, args):
        """Apply all optimizations to the model."""
        self.initializer_merger.move_input_tensor_to_attr(dag_model)
        self.initializer_merger.merge_initializers(dag_model)
        self.initializer_merger.convert_flat_to_reshape(dag_model)
        self.initializer_merger.remove_redundant_reshape(dag_model)
        self.fusion_optimizer.optimize(dag_model)

        dag_model.build_dependencies()

        if args.quant == "fp16":
            self.quantizer.quantize_to_fp16_selective(dag_model)
        elif args.quant == "int8":
            self.quantizer.quantize_to_int8_weight_only(dag_model)
