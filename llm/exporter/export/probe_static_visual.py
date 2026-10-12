"""验证：把 HF 视觉的三个 grid 预计算（interp/position_ids/cu_seqlens）在 trace 前算成常量
经 kwargs 注入，能否消掉 visual.onnx 里的 Loop/Sequence/CumSum，且数值与默认路径一致。"""

import os
import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    get_vision_cu_seqlens,
)

MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
EXPORT_IMG_SIZE = int(os.environ.get("EXPORT_IMG_SIZE", "224"))

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, attn_implementation="eager", torch_dtype=torch.float16)
model.eval()
visual = model.model.visual
cfg = visual.config
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

ps = cfg.patch_size
g = EXPORT_IMG_SIZE // ps
grid_thw = torch.tensor([[1, g, g]], dtype=torch.int32)
seq_len = int(grid_thw.prod())
row = cfg.in_channels * cfg.temporal_patch_size * ps * ps
pixel_values = torch.randn(seq_len, row, dtype=torch.float16)
print(f"[probe] patch={ps} grid={g}x{g} seq_len={seq_len} merge={cfg.spatial_merge_size} "
      f"num_pos_emb={cfg.num_position_embeddings}")


def precompute():
    """在 eager 模式下算好，返回值全部是常量张量。"""
    ii, iw = get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side=int(cfg.num_position_embeddings ** 0.5),
        mode=visual.interpolation_mode, align_corners=visual.interpolation_align_corners,
        spatial_merge_size=cfg.spatial_merge_size)
    pid = get_vision_position_ids(grid_thw, cfg.spatial_merge_size)
    cu = get_vision_cu_seqlens(grid_thw)
    return dict(interp_indices=ii, interp_weights=iw, position_ids=pid, cu_seqlens=cu)


with torch.no_grad():
    ref = visual(pixel_values, grid_thw)
    kw = precompute()
    print("[precompute] " + " ".join(f"{k}{tuple(v.shape)}={v.dtype}" for k, v in kw.items()))
    try:
        got = visual(pixel_values, grid_thw, **kw)
    except Exception as e:
        print(f"[FAIL] kwargs 被拒绝: {type(e).__name__}: {e}")
        raise SystemExit(1)

    a, b = ref.pooler_output, got.pooler_output
    da = torch.stack(ref.deepstack_features)
    db = torch.stack(got.deepstack_features)
    print(f"[compare] pooler maxdiff={(a - b).abs().max().item():.6g} bitwise={torch.equal(a, b)}")
    print(f"[compare] deepstack maxdiff={(da - db).abs().max().item():.6g} bitwise={torch.equal(da, db)}")
    ok = torch.equal(a, b) and torch.equal(da, db)
    print("[OK] kwargs 逃生口数值一致" if ok else "[DIFF] 不一致")
    if not ok:
        raise SystemExit(2)

# 导出看图里还有没有控制流
import onnx
from onnx import numpy_helper

class Vis(torch.nn.Module):
    def __init__(self, v, kw):
        super().__init__()
        self.v = v
        self.kw = kw

    def forward(self, pixel_values, grid_thw):
        kw = {k: v for k, v in self.kw.items()}
        out = self.v(pixel_values, grid_thw, **kw)
        return out.pooler_output, out.deepstack_features[0], out.deepstack_features[1], out.deepstack_features[2]


tmp = "/tmp/visual_static.onnx"
with torch.no_grad():
    torch.onnx.export(Vis(visual, precompute()), (pixel_values, grid_thw), tmp,
                      input_names=["pixel_values", "grid_thw"],
                      output_names=["image_features", "deepstack_features_0",
                                    "deepstack_features_1", "deepstack_features_2"],
                      opset_version=17, dynamo=False)
m = onnx.load(tmp, load_external_data=False)
bad = {"Loop", "SequenceEmpty", "SequenceAt", "SequenceInsert", "SplitToSequence",
       "ConcatFromSequence", "CumSum", "Scan", "If", "NonMaxSuppression", "OneHot", "Trilu"}
hits = [n.op_type for n in m.graph.node if n.op_type in bad]
print(f"[export] nodes={len(m.graph.node)} 控制流残留={hits or '无'}")
bool_inits = [i.name for i in m.graph.initializer if i.data_type == onnx.TensorProto.BOOL]
print(f"[export] bool initializers={len(bool_inits)} {bool_inits[:6]}")
for o in m.graph.output:
    print("  out", o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
print("[next] 用 onnxruntime 比 visual_static.onnx 与 eager 输出，再走 onnx2vkop 转换")
