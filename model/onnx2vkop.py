#!/bin/env python3

import os
import sys
import struct
import onnx
import numpy as np
from onnx import numpy_helper

class VkModel:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.initializers = {}

    def save_to_binary(self, file_path):
        with open(file_path, 'wb') as f:
            # Save inputs with shapes
            self._write_list_with_shapes(f, self.inputs)

            # Save outputs with shapes
            self._write_list_with_shapes(f, self.outputs)

            # Save nodes including attributes and input/output shapes
            f.write(struct.pack('I', len(self.nodes)))
            for node in self.nodes:
                self._write_string(f, node['op_type'])
                self._write_dict(f, node['attributes'])  # Attributes
                self._write_list_with_shapes(f, node['inputs'])
                self._write_list_with_shapes(f, node['outputs'])

            # Save initializers (name -> numpy array)
            f.write(struct.pack('I', len(self.initializers)))
            for name, arr in self.initializers.items():
                self._write_string(f, name)
                self._write_array(f, arr)

    @staticmethod
    def _write_string(f, s):
        b = s.encode('utf-8')
        f.write(struct.pack('I', len(b)))
        f.write(b)

    @staticmethod
    def _write_list_with_shapes(f, lst):
        f.write(struct.pack('I', len(lst)))
        for item in lst:
            VkModel._write_string(f, item['name'])
            VkModel._write_list(f, item['shape'])

    @staticmethod
    def _write_list(f, lst):
        f.write(struct.pack('I', len(lst)))
        for item in lst:
            if isinstance(item, int):
                f.write(struct.pack('I', item))  # For dimensions
            else:
                VkModel._write_string(f, item)  # For names

    @staticmethod
    def _write_dict(f, d):
        f.write(struct.pack('I', len(d)))
        for key, value in d.items():
            VkModel._write_string(f, key)
            if isinstance(value, str):
                f.write(b'\x00')  # Tag for string
                VkModel._write_string(f, value)
            elif isinstance(value, int):
                f.write(b'\x01')  # Tag for int
                f.write(struct.pack('q', value))
            elif isinstance(value, float):
                f.write(b'\x02')  # Tag for float
                f.write(struct.pack('d', value))

    @staticmethod
    def _write_array(f, arr):
        arr = np.ascontiguousarray(arr)
        dtype = arr.dtype.name
        shape = list(arr.shape)
        VkModel._write_string(f, dtype)
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
        data = arr.tobytes()
        f.write(struct.pack('Q', len(data)))
        f.write(data)

    @staticmethod
    def load_from_binary(file_path):
        model = VkModel()
        with open(file_path, 'rb') as f:
            model.inputs = VkModel._read_list_with_shapes(f)
            model.outputs = VkModel._read_list_with_shapes(f)

            num_nodes = struct.unpack('I', f.read(4))[0]
            for _ in range(num_nodes):
                op_type = VkModel._read_string(f)
                attributes = VkModel._read_dict(f)
                inputs = VkModel._read_list_with_shapes(f)
                outputs = VkModel._read_list_with_shapes(f)
                model.nodes.append({'op_type': op_type, 'attributes': attributes,
                                    'inputs': inputs, 'outputs': outputs})

            num_inits = struct.unpack('I', f.read(4))[0]
            for _ in range(num_inits):
                name = VkModel._read_string(f)
                dtype = VkModel._read_string(f)
                shape = [
                    struct.unpack('I', f.read(4))[0]
                    for _ in range(struct.unpack('I', f.read(4))[0])
                ]
                data_len = struct.unpack('Q', f.read(8))[0]
                data = f.read(data_len)
                arr = np.frombuffer(data, dtype=np.dtype(dtype))
                arr = arr.reshape(shape)
                model.initializers[name] = arr

        return model

    @staticmethod
    def _read_string(f):
        length = struct.unpack('I', f.read(4))[0]
        return f.read(length).decode('utf-8')

    @staticmethod
    def _read_list_with_shapes(f):
        count = struct.unpack('I', f.read(4))[0]
        return [
            {
            'name': VkModel._read_string(f),
            'shape': [
                struct.unpack('I', f.read(4))[0]
                for _ in range(struct.unpack('I', f.read(4))[0])
            ]
            }
            for _ in range(count)
        ]

    @staticmethod
    def _read_dict(f):
        count = struct.unpack('I', f.read(4))[0]
        return {
            VkModel._read_string(f): VkModel._read_value(f)
            for _ in range(count)
        }

    @staticmethod
    def _read_value(f):
        tag = f.read(1)
        if tag == b'\x00':
            return VkModel._read_string(f)
        if tag == b'\x01':
            return struct.unpack('q', f.read(8))[0]
        if tag == b'\x02':
            return struct.unpack('d', f.read(8))[0]
        return "unknown"


def parse_onnx_model(onnx_path):
    model = onnx.load(onnx_path)
    vk_model = VkModel()

    graph = model.graph

    # Inputs with shapes
    for inp in graph.input:
        tensor_type = inp.type.tensor_type
        shape_dims = [
            dim.dim_value if dim.HasField("dim_value") else 1
            for dim in tensor_type.shape.dim
        ]
        vk_model.inputs.append({'name': inp.name, 'shape': shape_dims})

    # Outputs with shapes
    for out in graph.output:
        tensor_type = out.type.tensor_type
        shape_dims = [
            dim.dim_value if dim.HasField("dim_value") else 1
            for dim in tensor_type.shape.dim
        ]
        vk_model.outputs.append({'name': out.name, 'shape': shape_dims})
    # Nodes with attributes and shapes of inputs/outputs
    for node in graph.node:
        attributes = {}
        for attr in node.attribute:
            if attr.HasField('i'):
                attributes[attr.name] = attr.i
            elif attr.HasField('f'):
                attributes[attr.name] = attr.f
            elif attr.HasField('s'):
                attributes[attr.name] = attr.s.decode('utf-8')
            # Add more types if necessary

        inputs_with_shape = []
        for input_name in node.input:
            for inp in graph.input:
                if inp.name == input_name:
                    tensor_type = inp.type.tensor_type
                    shape_dims = [
                        dim.dim_value if dim.HasField("dim_value") else 1
                        for dim in tensor_type.shape.dim
                    ]
                    inputs_with_shape.append(
                        {'name': input_name, 'shape': shape_dims}
                    )
                    break

        outputs_with_shape = []
        for output_name in node.output:
            for out in graph.output:
                if out.name == output_name:
                    tensor_type = out.type.tensor_type
                    shape_dims = [
                        dim.dim_value if dim.HasField("dim_value") else 1
                        for dim in tensor_type.shape.dim
                    ]
                    outputs_with_shape.append(
                        {'name': output_name, 'shape': shape_dims}
                    )
                    break

        vk_model.nodes.append({
            'op_type': node.op_type,
            'attributes': attributes,
            'inputs': inputs_with_shape,
            'outputs': outputs_with_shape
        })

    # Initializers (parameters)
    for initializer in graph.initializer:
        name = initializer.name
        # shape = tuple(initializer.dims)
        # data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
        # raw_data = initializer.raw_data
        arr = numpy_helper.to_array(initializer)
        vk_model.initializers[name] = arr

    return vk_model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx2vkop.py <onnx_model_path>")
        sys.exit(1)

    onnx_model_path = sys.argv[1]
    output_bin_path = os.path.splitext(onnx_model_path)[0] + ".bin"

    print("Parsing ONNX model...")
    onnxmodel = parse_onnx_model(onnx_model_path)

    print("Saving to binary format...")
    onnxmodel.save_to_binary(output_bin_path)

    print("Loading back from binary...")
    loaded_model = VkModel.load_from_binary(output_bin_path)

    print("\nInputs:")
    for inp in loaded_model.inputs:
        print(f"  Name: {inp['name']}, Shape: {inp['shape']}")

    print("\nOutputs:")
    for out in loaded_model.outputs:
        print(f"  Name: {out['name']}, Shape: {out['shape']}")

    print("\nNodes:")
    for node in loaded_model.nodes:
        print(f"  OpType: {node['op_type']}, Attributes: {node['attributes']}")
        print("    Inputs:")
        for input_info in node['inputs']:
            print(f"      Name: {input_info['name']}, Shape: {input_info['shape']}")
        print("    Outputs:")
        for output_info in node['outputs']:
            print(f"      Name: {output_info['name']}, Shape: {output_info['shape']}")

    print("\nInitializers:")
    for name, arr in loaded_model.initializers.items():
        print(f"  Name: {name}, Shape: {arr.shape}, DType: {arr.dtype}")
