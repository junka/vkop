"""一致性测试用例构造：合成 PIL 图、prompt 模板、CASES 列表。

供 tests/ 下的 pytest 用例复用，也可被 ad-hoc 脚本 import。
所有合成图像尺寸固定 224×224（= visual.onnx 导出尺寸，因 grid_thw 被常量折叠）。
"""
import os
from PIL import Image, ImageDraw, ImageFont

PROMPT_TMPL = "<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
IMG_TOK = "<|vision_start|><|image_pad|><|vision_end|>"

def text_body(q):
    return q

def img_body(q, image):
    return f"{IMG_TOK}{q}"

def solid_color_image(color, size=(224, 224)):
    return Image.new("RGB", size, color=color)

def text_image(text, size=(224, 224), bg="white", fg="black"):
    """画文字到纯色背景，做 OCR 测试。"""
    img = Image.new("RGB", size, color=bg)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size[0] - w) / 2 - bbox[0], (size[1] - h) / 2 - bbox[1]),
           text, fill=fg, font=font)
    return img

# 用例：(id, kind, payload)
#   kind="text"  payload = question 字符串
#   kind="image" payload = (image, question)
# 覆盖业界通用场景：纯文本 / 图像理解 / OCR / 边界尺寸
CASES = [
    # --- 纯文本：计数 / 翻译 / 常识 ---
    ("text_count",     "text", "Count from 1 to 5."),
    ("text_translate", "text", "Translate 'hello' to Chinese. One word."),
    ("text_common",    "text", "What is the capital of France? One word."),
    # --- 图像理解：颜色识别 ---
    ("img_color_red",   "image", (solid_color_image((220, 20, 20)), "What color is this image? One word.")),
    ("img_color_green", "image", (solid_color_image((20, 180, 20)), "What color is this image? One word.")),
    ("img_color_blue",  "image", (solid_color_image((20, 20, 200)), "What color is this image? One word.")),
    # --- OCR：图中文字/数字识别 ---
    ("ocr_number_42", "image", (text_image("42"), "What number is shown? One word.")),
    ("ocr_word_HI",   "image", (text_image("HI", size=(224, 224)), "What text is shown? One word.")),
    # --- 边界尺寸：不同纯色图、长文本 ---
    ("img_color_yellow", "image", (solid_color_image((200, 200, 40)), "What color is this image? One word.")),
    ("text_long",         "text", "List three primary colors, comma separated."),
]

def case_prompt(case):
    """从用例构造完整 prompt 字符串。"""
    cid, kind, payload = case
    if kind == "text":
        return PROMPT_TMPL.format(body=text_body(payload)), None
    image, q = payload
    return PROMPT_TMPL.format(body=img_body(q, image)), image
