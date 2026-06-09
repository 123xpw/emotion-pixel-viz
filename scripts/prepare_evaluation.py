"""
生成强迫选择评估所需的资料：
  1. outputs/evaluation_panels/diary_N_panel.png  — A/B/C 对比面板（用于 Google Form）
  2. outputs/answer_key.json                       — 条件-字母映射（用于后续统计）
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# 固定打乱方案（Latin-square 均衡，每条件在 A/B/C 各位置出现 3-4 次）
# 生成后不再修改，否则已填写问卷的答案会失效
# ─────────────────────────────────────────────────────────────────────────────
SHUFFLE = {
    "diary_1":  {"A": "full",     "B": "partial",  "C": "baseline"},
    "diary_2":  {"A": "partial",  "B": "baseline", "C": "full"},
    "diary_3":  {"A": "baseline", "B": "full",     "C": "partial"},
    "diary_4":  {"A": "partial",  "B": "full",     "C": "baseline"},
    "diary_5":  {"A": "baseline", "B": "partial",  "C": "full"},
    "diary_6":  {"A": "full",     "B": "baseline", "C": "partial"},
    "diary_7":  {"A": "full",     "B": "partial",  "C": "baseline"},
    "diary_8":  {"A": "baseline", "B": "full",     "C": "partial"},
    "diary_9":  {"A": "partial",  "B": "baseline", "C": "full"},
    "diary_10": {"A": "baseline", "B": "partial",  "C": "full"},
}

CONDITION_DIRS = {
    "full":     BASE_DIR / "outputs" / "condition_full",
    "partial":  BASE_DIR / "outputs" / "condition_partial",
    "baseline": BASE_DIR / "outputs" / "condition_baseline",
}

IMG_SIZE    = 400   # 每张子图缩放尺寸（像素）
PADDING     = 24    # 图间距
LABEL_H     = 48    # 字母标签区高度
BG_COLOR    = (245, 245, 245)
LABEL_COLOR = (30, 30, 30)
BORDER_COLOR = (200, 200, 200)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def create_panel(diary_id: str, output_dir: Path) -> None:
    mapping = SHUFFLE[diary_id]
    imgs = {}
    for letter, condition in mapping.items():
        path = CONDITION_DIRS[condition] / f"{diary_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"找不到图片: {path}")
        imgs[letter] = Image.open(path).convert("RGB")

    total_w = PADDING + 3 * (IMG_SIZE + PADDING)
    total_h = PADDING + IMG_SIZE + LABEL_H + PADDING

    panel = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw  = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype(FONT_PATH, 32)
    except Exception:
        font = ImageFont.load_default()

    for i, letter in enumerate(["A", "B", "C"]):
        x = PADDING + i * (IMG_SIZE + PADDING)
        y = PADDING

        img_r = imgs[letter].resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        panel.paste(img_r, (x, y))

        # 细边框
        draw.rectangle([x - 1, y - 1, x + IMG_SIZE, y + IMG_SIZE],
                       outline=BORDER_COLOR, width=1)

        # 字母标签居中
        lx = x + IMG_SIZE // 2
        ly = y + IMG_SIZE + LABEL_H // 2
        draw.text((lx, ly), letter, fill=LABEL_COLOR, font=font, anchor="mm")

    out_path = output_dir / f"{diary_id}_panel.png"
    panel.save(out_path)
    print(f"  ✓ {out_path.name}")


def save_answer_key() -> None:
    # 正向：diary → {A/B/C → condition}
    # 反向：diary → {condition → A/B/C}（统计时用）
    reverse = {
        diary_id: {cond: letter for letter, cond in mapping.items()}
        for diary_id, mapping in SHUFFLE.items()
    }
    key = {"shuffle": SHUFFLE, "condition_to_letter": reverse}
    out = BASE_DIR / "outputs" / "answer_key.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    print(f"  ✓ answer_key.json")


def main():
    panel_dir = BASE_DIR / "outputs" / "evaluation_panels"
    panel_dir.mkdir(exist_ok=True)

    print("生成对比面板...")
    for diary_id in sorted(SHUFFLE.keys(), key=lambda x: int(x.split("_")[1])):
        create_panel(diary_id, panel_dir)

    print("\n保存答案密钥...")
    save_answer_key()

    print("\n完成。文件位置：")
    print(f"  对比面板：{panel_dir}")
    print(f"  答案密钥：{BASE_DIR / 'outputs' / 'answer_key.json'}")


if __name__ == "__main__":
    main()
