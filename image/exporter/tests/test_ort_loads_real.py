"""真权重图能不能被 ORT 载入并跑起来 —— 只做载入/绑定/一次小 shape 的前向。

这一步是 `dit_prefill.onnx` 的第一道真关卡：上一版用 `onnx.save_model` 合出来的 external
data 是坏账，ORT 在解析阶段就直接拒载（`GetExternalDataInfo ... external data size mismatch`），
而 python 侧 `onnx.load` 和我们一起写的 `ext_check` 都没查出问题。现在换成流式合并，
所以"ORT 读得进去"必须由 ORT 自己作证，不能只看 `ext_check` 绿。

为什么不在这个脚本里比数值：比数值要同时持有 torch 那份 14.2 GB 和 ORT 那份 13.8 GB
（=28 GB，这台 36 GB 的机器上会打到 swap），所以真权重的逐张量对齐要走**两个进程 + 中间量
落盘**。本脚本只花一份权重的内存。

    /Users/doudou/qi21-env/bin/python tests/test_ort_loads_real.py [prefill|decode]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from probe_dit import cur_rss_gb  # noqa: E402  和内存探针同一把尺子（ps 的 KB -> GB）

CFG_PATH = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1/snapshots/master"
                "/transformer/config.json")
# 故意取小到 S=32：这一测只关心"载入 + 绑定 + 算得动"，激活尺寸越小越好，
# 序列轴是 dynamic 的，所以换尺寸不用重导图。
S = {"prefill": 32, "decode": 32}


def rss_gb():
    return cur_rss_gb()


def main(mode):
    hd = json.loads(CFG_PATH.read_text())["attention_head_dim"]
    path = HERE / f"dit_{mode}.onnx"
    bin_path = HERE / f"dit_{mode}.weights.bin"
    m = onnx.load(str(path), load_external_data=False)
    print(f"[{mode}] nodes={len(m.graph.node)} graph initializers={len(m.graph.initializer)} "
          f"bin={bin_path.stat().st_size:,}B")

    t0, r0 = time.time(), rss_gb()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"[load] {time.time() - t0:.1f}s RSS {r0:.2f} -> {rss_gb():.2f} GB")

    s = S[mode]
    # 符号轴 -> 具体尺寸。`kv_len` 必须是 prefix+target 两段之和（decode 的 bias 覆盖整条
    # 联合序列），其余轴都取 s；图是 dynamic 的，换分辨率不用重导。
    symbols = {"prefix_len": s, "target_len": s, "kv_len": 2 * s}
    # dtype 从 graph proto 的 elem_type 取，不从 ORT 的 `i.type` 字符串猜：那个串在 fp16 上
    # 不一定写 "fp16"，猜错过一次（`Unexpected input data type. Actual: (tensor(float))`）。
    np_dt = {vi.name: onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
             for vi in m.graph.input}
    feed = {}
    for i in sess.get_inputs():
        dims = tuple(symbols.get(d, d) for d in i.shape)
        dt = np_dt[i.name]
        # 随机数只喂给真正参与计算的输入；KV/bias/timestep 给零，这样 NaN 只可能是"算出来"的，
        # 下面那条 isfinite 断言才有意义。
        if i.name.startswith(("past_kv", "attention_bias", "timestep")):
            arr = np.zeros(dims, dtype=dt)
        elif i.name in ("cos", "sin"):
            assert dims[1] == hd, (dims, hd)  # rope 表的宽度就是 config 里的 head_dim
            arr = np.tile(np.cos(np.arange(dims[0])[:, None] * 0.01), (1, hd)).astype(dt)
        else:
            arr = (np.random.randn(*dims) * 0.02).astype(dt)
        feed[i.name] = arr
        print(f"  in  {i.name:<16} {dt} {list(i.shape)} -> {arr.shape}")
    t0 = time.time()
    outs = sess.run(None, feed)
    print(f"[run] {time.time() - t0:.1f}s RSS {rss_gb():.2f} GB, "
          f"{len(outs)} 输出，|max| = " + ", ".join(f"{np.abs(o).max():.4g}" for o in outs[:3])
          + (" ..." if len(outs) > 3 else ""))
    for o, name in zip(outs, [x.name for x in sess.get_outputs()]):
        assert np.isfinite(o).all(), f"{name} 出 NaN/Inf 了"
    print(f"[ok] ORT 载入并跑通 {mode}（fp16 external data 读得进去，输出全有限）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prefill")
