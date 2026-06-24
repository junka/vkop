#!/usr/bin/env python3
"""
ResNet ONNX Runtime Inference Script

Usage:
    python resnet.py <model.onnx> <image.jpg> [--top K] [--labels labels.txt]

调试中间节点输出：
    # 1) 先列出模型所有节点，找到想调试的节点名/张量名
    python resnet.py <model.onnx> --list-nodes            # 列出全部节点
    python resnet.py <model.onnx> --list-nodes Conv       # 按 op 类型过滤

    # 2) 把指定节点/张量暴露为输出后推理，自动打印其统计信息
    python resnet.py <model.onnx> <image.jpg> --debug-node <name>
    python resnet.py <model.onnx> <image.jpg> --debug-node <name> --save-debug
"""

import argparse
import numpy as np
from PIL import Image 
import onnx
from onnx import TensorProto
import onnxruntime as ort
import sys
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

def load_labels(label_file):
    if not os.path.exists(label_file):
        print(f"Warning: Label file {label_file} not found, using indices.")
        return [f"Class_{i}" for i in range(1000)]
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path):
    # 用 PIL 读取并转为 RGB
    img = Image.open(image_path).convert('RGB')
    # 使用 BILINEAR 插值（默认也是 BILINEAR，但明确指定）
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    # 转为 numpy 数组，并归一化
    img = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    # 转成 NCHW
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def resize_uint8_hwc(image_path, size=INPUT_SIZE):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.array(img, dtype=np.uint8))

def pack_rgba_hwc(nchw_arr):
    """把 NCHW [1,3,H,W] float32 转成 HWC-RGBA float32（A 通道补 0），
    复刻 C++ 端 copyToGPUImage 前的 normalized_data 内存布局，可逐字节对比。"""
    hwc = nchw_arr[0].transpose(1, 2, 0)  # [H,W,3]
    h, w, _ = hwc.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = hwc
    return np.ascontiguousarray(rgba)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def find_tensors(model, pattern):
    """根据 op 类型、节点名或张量名查找张量名，返回 (tensor_name, producer_node) 列表。
    pattern 可以是 op 类型（如 'Conv'/'Relu'/'Gemm'）、节点名，或张量名。"""
    matches = []
    for node in model.graph.node:
        is_op = pattern.lower() == node.op_type.lower()
        is_node = pattern == node.name
        if is_op or is_node:
            # 优先输出节点的首个输出张量
            for out in node.output:
                matches.append((out, node))
            break
    # 也支持直接按张量名匹配（包括 graph 的输入/输出/initializer/中间张量）
    if not matches:
        graph_tensors = set(model.graph.input) | set(model.graph.output)
        names_in_graph = {t.name for t in graph_tensors}
        names_in_graph |= {t.name for t in model.graph.initializer}
        for node in model.graph.node:
            for out in node.output:
                names_in_graph.add(out)
        if pattern in names_in_graph:
            producer = None
            for node in model.graph.node:
                if pattern in node.output:
                    producer = node
                    break
            matches.append((pattern, producer))
    return matches


def expose_outputs(model, tensor_names):
    """把指定的中间张量追加为模型输出，方便 onnxruntime 一次性把它们读出来。
    会克隆 graph 的 value_info 中的形状信息（如果存在），否则用空 TensorProto。"""
    graph = model.graph
    existing = {o.name for o in graph.output}
    info_map = {v.name: v for v in graph.value_info}
    info_map.update({i.name: i for i in graph.input})
    info_map.update({o.name: o for o in graph.output})

    for name in tensor_names:
        if name in existing:
            continue
        if name in info_map:
            graph.output.append(info_map[name])
        else:
            # 未知形状，给一个空描述，onnxruntime 仍可推理得到其真实形状
            graph.output.append(onnx.ValueInfoProto(name=name))

    # 用最新 IR 版本重新序列化，确保 onnxruntime 能加载修改后的图
    model = onnx.shape_inference.infer_shapes(model) if _can_infer(model) else model
    onnx.checker.check_model(model)
    return model


def _can_infer(model):
    try:
        return model.ir_version >= 3
    except Exception:
        return True


def list_nodes(model_path, filter_op=None):
    """打印模型的所有节点，方便定位要调试的张量。"""
    model = onnx.load(model_path)
    print(f"\n========== Nodes in {os.path.basename(model_path)} ==========")
    print(f"{'idx':>4}  {'op_type':<14} {'node_name':<40} outputs")
    count = 0
    for i, node in enumerate(model.graph.node):
        if filter_op and filter_op.lower() not in node.op_type.lower() \
                and filter_op not in node.name \
                and filter_op not in ','.join(node.output):
            continue
        outs = ', '.join(node.output) if node.output else '(none)'
        name = node.name if node.name else '(unnamed)'
        print(f"{i:>4}  {node.op_type:<14} {name:<40} {outs}")
        count += 1
    print(f"-- {count} node(s)" + (f" matched '{filter_op}'" if filter_op else ""))


def make_debug_session(model_path, debug_names):
    """加载模型，把 debug_names 对应的张量暴露为额外输出，返回可推理的 session。"""
    model = onnx.load(model_path)
    targets = []
    for name in debug_names:
        found = find_tensors(model, name)
        if not found:
            raise ValueError(
                f"未找到与 '{name}' 匹配的张量/节点。"
                f"可用 --list-nodes 查看模型节点列表。")
        for tensor_name, _ in found:
            targets.append(tensor_name)
    targets = list(dict.fromkeys(targets))  # 去重保序
    model = expose_outputs(model, targets)
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=['CPUExecutionProvider'])
    return sess, targets


def describe_tensor(name, arr, save=False, save_dir='.'):
    """打印张量的形状/统计信息，可选保存到 .npy。"""
    print(f"\n---- [debug] {name} ----")
    print(f"  shape: {arr.shape}  dtype: {arr.dtype}")
    print(f"  min: {np.nanmin(arr):.6f}  max: {np.nanmax(arr):.6f}  "
          f"mean: {np.nanmean(arr):.6f}  std: {np.nanstd(arr):.6f}")
    flat = arr.reshape(-1)
    if flat.size > 0:
        print(f"  first 32 values: {flat[:32]}")
    if save:
        safe = name.replace('/', '_').replace(':', '_')
        path = os.path.join(save_dir, f"debug_{safe}.npy")
        np.save(path, arr)
        print(f"  saved -> {path}")


def dump_model_input(model_path, image_path, input_tensor):
    """把网络输入张量 dump 成多种布局，方便和 C++ 端逐字节对比。

    保存到模型同目录下：
      py_input_nchw.npy    NCHW [1,3,H,W] float32（onnxruntime 直接吃的）
      py_input_rgba.npy    HWC-RGBA float32，A 通道补 0（复刻 C++ copyToGPUImage 内存布局）
      py_input_hwc_u8.npy  resize 后未归一化的 HWC RGB uint8（复刻 stbir 输出）

    同时打印统计信息和 NCHW 前若干值，方便和 C++ 日志里看到的 input 对比。
    """
    save_dir = os.path.dirname(os.path.abspath(model_path))

    # 1) NCHW（网络真实输入）
    nchw = np.ascontiguousarray(input_tensor, dtype=np.float32)
    np.save(os.path.join(save_dir, "py_input_nchw.npy"), nchw)

    # 2) RGBA HWC（C++ 进 GPU 前的内存布局）
    rgba = pack_rgba_hwc(nchw)
    np.save(os.path.join(save_dir, "py_input_rgba.npy"), rgba)

    # 3) resize 后未归一化 HWC uint8（直接对比 stbir 输出）
    hwc_u8 = resize_uint8_hwc(image_path, nchw.shape[-1])
    np.save(os.path.join(save_dir, "py_input_hwc_u8.npy"), hwc_u8)

    print("\n========== Model Input Dump ==========")
    print(f"saved to: {save_dir}")
    print(f"  py_input_nchw.npy    shape={nchw.shape} dtype={nchw.dtype}")
    print(f"  py_input_rgba.npy    shape={rgba.shape} dtype={rgba.dtype}")
    print(f"  py_input_hwc_u8.npy  shape={hwc_u8.shape} dtype={hwc_u8.dtype}")

    print("\n--- [input] NCHW 通道0 前 32 个值（C/H/W=0/0/0/0..）---")
    flat = nchw.reshape(-1)
    print(f"  {flat[:32]}")
    print(f"  nchw: min={nchw.min():.6f} max={nchw.max():.6f} "
          f"mean={nchw.mean():.6f} std={nchw.std():.6f}")

    # 按 HWC RGBA 布局打印，和 C++ 端 normalized_data[(i*4)+c] 对齐
    # C++ 端若是逐像素扫描，最先存的是 HWC RGBA；这里打印 channel0 的前 8 个像素
    print("\n--- [input] RGBA-HWC channel0 前 8 个像素值 ---")
    print(f"  {[float(rgba.flatten()[i*4]) for i in range(8)]}")

    print("\n--- [input] resize 后 HWC uint8 RGB 前 8 像素（对比 stbir 输出）---")
    print(f"  {hwc_u8.reshape(-1)[:24].tolist()}")


def inference(model_path, image_path, top_k=10, label_file=None,
              debug_names=None, save_debug=False, dump_input=False):
    # 使用 CPU 执行
    if debug_names:
        sess, debug_targets = make_debug_session(model_path, debug_names)
    else:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        debug_targets = []
    input_name = sess.get_inputs()[0].name

    input_tensor = preprocess_image(image_path)

    if dump_input:
        dump_model_input(model_path, image_path, input_tensor)
    outputs = sess.run(None, {input_name: input_tensor})

    # 最后一个输出保持是原始模型输出（logits）
    output_names = [o.name for o in sess.get_outputs()]
    logits_out = outputs[0][0]

    if debug_targets:
        print("\n========== Debug Tensors ==========")
        # outputs 与 sess.get_outputs() 顺序一致
        for name, arr in zip(output_names, outputs):
            if name in debug_targets:
                describe_tensor(name, arr, save=save_debug,
                                save_dir=os.path.dirname(os.path.abspath(model_path)))

    probs = softmax(logits_out)

    top_indices = np.argsort(probs)[::-1][:top_k]

    if label_file:
        labels = load_labels(label_file)
    else:
        default_label = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
        labels = load_labels(default_label) if os.path.exists(default_label) else [f"Class_{i}" for i in range(probs.shape[0])]

    print(f"\n========== Top-{top_k} Predictions ==========")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {labels[idx] if idx < len(labels) else f'Class_{idx}'} : {probs[idx]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="ResNet ONNX Runtime 推理 & 中间节点调试")
    parser.add_argument("model", help="ONNX model file")
    parser.add_argument("image", nargs="?", help="Input image (JPG/PNG)。--list-nodes 时可省略")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--labels", help="Labels file (one per line)")
    parser.add_argument("--list-nodes", nargs="?", const="", metavar="FILTER",
                        help="列出模型节点（可带过滤：op类型/节点名/张量名）后退出，不推理")
    parser.add_argument("--debug-node", dest="debug_nodes", action="append", default=[],
                        help="把指定节点/张量暴露为输出并打印（可多次指定）")
    parser.add_argument("--save-debug", action="store_true",
                        help="把 --debug-node 的张量保存为 .npy")
    parser.add_argument("--dump-input", action="store_true",
                        help="打印并保存网络输入张量（多种布局），仅做对比不推理")
    args = parser.parse_args()

    try:
        if args.list_nodes is not None:
            list_nodes(args.model, args.list_nodes or None)
            return

        if not args.image:
            parser.error("推理需要 image 参数（或使用 --list-nodes / --dump-input）")

        if args.dump_input:
            # 只 dump 输入，不跑推理
            import onnx as _onnx
            input_tensor = preprocess_image(args.image)
            dump_model_input(args.model, args.image, input_tensor)
            return

        inference(args.model, args.image, args.top, args.labels,
                  debug_names=args.debug_nodes or None,
                  save_debug=args.save_debug)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()