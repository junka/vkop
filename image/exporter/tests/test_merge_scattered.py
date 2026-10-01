"""`merge_scattered` 的字节账：合成一批**尺寸不凑巧**的张量，走一遍合并再逐字节读回。

只有真权重（14.2 GB）才会踩到 torch 的**散文件**那条分支（legacy 导出器只在整份模型超过
2 GB 时才自动外置权重），而这台机器上没法每次改动都拿真权重试错 —— 合并这一步写歪一个
offset 就是静默的数值错误，所以这里用合成图把两条来源（inline `raw_data` / 散文件）、同一
份内容挂在两处、以及 64B 对齐填充都钉住。

尺寸故意取 1026B 这种**不是 ALIGN(64) 整数倍**的：真图上所有权重都是 64 的倍数，填充恒为 0，
拿真图测不出"offset 少加一次填充"这种 bug（第一次合并坏掉时 ORT 报的
`Computed size: 33554432, external_data.length: 100663296` 就是这一类错位）。

    /Users/doudou/qi21-env/bin/python tests/test_merge_scattered.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnx.external_data_helper import _get_all_tensors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from qi21_export_onnx import ALIGN, ext_check, merge_scattered, _point_at  # noqa: E402

LOC = "sim.weights.bin"


def fp16(n):
    """n 个可辨识的 fp16 值：字节内容逐位唯一，读回时对不上就是错位。"""
    return (np.arange(n, dtype=np.float32) % 512).astype(np.float16)


def tensor(name, values):
    t = TensorProto()
    t.name, t.data_type = name, TensorProto.FLOAT16
    t.dims.extend([len(values)])
    t.raw_data = values.tobytes()
    return t


def main():
    # 三种够大的：inline 的非对齐、散文件的非对齐、正好对齐的；再加一个够小的（该留在 proto 里）。
    inline_odd = tensor("w_inline_odd", fp16(513))      # 1026B = 16*64 + 2
    scatter_odd = tensor("w_scatter_odd", fp16(700))    # 1400B = 21*64 + 56
    shared = tensor("w_shared_even", fp16(1024))        # 2048B，正好对齐
    small = tensor("axes", fp16(2))                     # 4B < INLINE_KEEP
    for t in (scatter_odd, shared):
        Path(f"{t.name}.bin").write_bytes(t.raw_data)
        # 模拟 torch 散文件形态：数据在只属于它的那个文件里，proto 只剩引用。
        _point_at(t, f"{t.name}.bin", 0, len(t.raw_data))
        assert not t.raw_data

    # `shared` 同时挂在 initializer 和 node attribute 上。protobuf 赋值时**拷贝**消息，所以图里
    # 是两个内容相同、`id()` 不同的 TensorProto，合并会把同一份数据写**两遍**、各占一段 offset。
    # 真图上每个权重只有一个 proto，所以"按内容去重"没有收益、只会让区间账变得不可复现。
    m = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Constant", [], ["c0"], value=scatter_odd),
             onnx.helper.make_node("Constant", [], ["c1"], value=small),
             onnx.helper.make_node("Constant", [], ["c2"], value=shared)],
            "g", inputs=[],
            outputs=[onnx.helper.make_tensor_value_info("c0", TensorProto.FLOAT16, [700])],
            initializer=[inline_odd, shared]),
        opset_imports=[onnx.helper.make_opsetid("", 17)])
    # pip 的 onnx 默认写最新的 ir_version，ORT 只认到它编译时那个版本 —— 合成图自己定一个。
    m.ir_version = 8

    # 合并是**原地**改 proto，改完就再也拿不到原始字节了，所以先按遍历顺序快照。
    # 注意不能按 `id(t)` 索引：protobuf 每次访问 repeated message 都新建一个 python 包装对象，
    # 同一个张量两轮遍历拿到的 `id()` 就不一样 —— 只有**顺序**是稳定的（合并不改图结构）。
    snapshot = [(t.name, t.raw_data or Path(
        {kv.key: kv.value for kv in t.external_data}["location"]).read_bytes())
        for t in _get_all_tensors(m)]
    order = [name for name, _ in snapshot]
    assert order == ["w_inline_odd", "w_shared_even", "w_scatter_odd", "axes",
                     "w_shared_even"], order

    written, nbytes = merge_scattered(m, LOC)
    assert (written, nbytes) == (4, 1026 + 2048 + 1400 + 2048), (written, nbytes)

    body = Path(LOC).read_bytes()
    offset, merged = 0, []
    for (name, data), t in zip(snapshot, _get_all_tensors(m)):
        assert t.name == name, (t.name, name)  # 顺序真的没变
        d = {kv.key: kv.value for kv in t.external_data}
        if t.data_location != TensorProto.EXTERNAL:
            assert (name, d) == ("axes", {}), name
            continue
        o, ln = int(d["offset"]), int(d["length"])
        assert (o, ln) == (offset, len(data)), (name, o, offset, ln, len(data))
        assert o % ALIGN == 0, (name, o)
        assert body[o:o + ln] == data, f"{name}: 读回的字节和原始数据不一致"
        merged.append(name)
        offset = o + ln + (-ln) % ALIGN  # 填充只可能出现在张量之后
    assert merged == order[:3] + order[4:], merged
    assert len(body) == offset, (len(body), offset)  # 文件末尾 = 最后一个张量 + 它的填充
    assert not Path("w_scatter_odd.bin").exists(), "散文件该被删掉"

    onnx.save(m, "sim.onnx")
    # 合并后自己那套账要能过（它查的正是 ORT 赖以取数的 offset/length 区间）。
    ext_check(Path("sim.onnx"), LOC, onnx.load("sim.onnx", load_external_data=False))
    # 真读一遍：ORT 按 external_data 取数据，坏账在这里就炸。合成图里有几个 initializer 没接到
    # 节点上，ORT 会打 "Removing initializer ..." 的 warning —— 那是图骨架的问题，与字节账无关。
    sess = ort.InferenceSession("sim.onnx", providers=["CPUExecutionProvider"])
    got = np.asarray(sess.run(None, {})[0], dtype=np.float32)
    assert np.array_equal(got, np.asarray(fp16(700), dtype=np.float32)), "ORT 取到的数据不对"
    print(f"[verdict] PASS  ({len(merged)} 个张量 / {len(body):,}B 文件，"
          f"offset 与填充逐字节对得上，ORT 读回逐位相同)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        prev = os.getcwd()
        os.chdir(td)  # 散文件和 loc 都是**相对 CWD** 的裸文件名，和真导出保持一致
        try:
            main()
        finally:
            os.chdir(prev)
