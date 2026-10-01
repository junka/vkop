"""把上游 diffusers 的 Qwen-Image-2.1 模块原样搬进本目录，只改写包内相对导入。

参考实现必须是**上游代码本身**，不是我复述的版本——复述错了就测不出错。这些文件在 PyPI
发布的 diffusers 里还没有（只有 main 分支有），所以按文件名从仓库取回后放在这里，用绝对
导入的形式独立成模块。

    /Users/doudou/qi21-env/bin/python reference/make_reference.py <源目录>
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 相对导入 -> 绝对导入。按前缀长度从长到短替换，免得 `...utils` 先吃掉 `...utils.peft_utils`。
SUBS = {
    "...configuration_utils": "diffusers.configuration_utils",
    "...loaders": "diffusers.loaders",
    "...utils": "diffusers.utils",
    "...utils.peft_utils": "diffusers.utils.peft_utils",
    "...utils.torch_utils": "diffusers.utils.torch_utils",
    "...utils.accelerate_utils": "diffusers.utils.accelerate_utils",
    "..attention": "diffusers.models.attention",
    "..attention_dispatch": "diffusers.models.attention_dispatch",
    "..cache_utils": "diffusers.models.cache_utils",
    "..embeddings": "diffusers.models.embeddings",
    "..modeling_outputs": "diffusers.models.modeling_outputs",
    "..modeling_utils": "diffusers.models.modeling_utils",
    "..normalization": "diffusers.models.normalization",
    "..activations": "diffusers.models.activations",
    ".vae": "diffusers.models.autoencoders.vae",
}


def convert(src_path: Path, out_name: str) -> None:
    text = src_path.read_text()
    for k in sorted(SUBS, key=len, reverse=True):
        text = text.replace(f"from {k} import", f"from {SUBS[k]} import")
    text = text.replace("from ..cache_utils import", "from diffusers.models.cache_utils import")
    leftovers = re.findall(r"^from (\.+)(\S*) import.*$", text, re.M)
    assert not leftovers, (out_name, leftovers)
    header = (f"# 上游 diffusers `{src_path.name}` 的原样副本，仅把包内相对导入改成绝对导入。\n"
              f"# 由 reference/make_reference.py 生成，请勿手改；改动会掩盖与上游实现的偏差。\n")
    (HERE / out_name).write_text(header + text)
    print(f"[reference] {src_path} -> {HERE / out_name} "
          f"({len(text.splitlines())} 行, 相对导入已全部改写)")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    convert(root / "transformer_qwenimage21.py", "qi21_transformer.py")
    convert(root / "autoencoder_kl_qwenimage21.py", "qi21_vae.py")
