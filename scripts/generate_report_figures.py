"""
生成报告配图，所有数据均来自实测结果（clip_scores.csv + 真实生成图像）。
产出：
  outputs/figure_condition_comparison.png  —— diary_1 三条件并排对比（含 CLIP 分）
  outputs/figure_clip_heatmap.png          —— 10篇日记 × 3条件 CLIP 分热力图
"""

import csv
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = Path(__file__).parent.parent
OUTPUTS = BASE_DIR / "outputs"
CONDITIONS = ["condition_baseline", "condition_partial", "condition_full"]
LABELS = {"condition_baseline": "Baseline", "condition_partial": "Partial", "condition_full": "Full"}
COLORS = {"condition_baseline": "#5b9bd5", "condition_partial": "#ed7d31", "condition_full": "#70ad47"}

# ── 读取实测 CLIP 分 ──────────────────────────────────────────────
def load_clip_scores():
    scores = {}  # {diary_id: {condition: score}}
    with open(OUTPUTS / "clip_scores.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d, c, s = row["diary"], row["condition"], float(row["clip_score"])
            scores.setdefault(d, {})[c] = s
    return scores


# ── Figure 1：diary_1 三条件并排 ──────────────────────────────────
def make_condition_comparison(scores, diary_id="diary_1"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(
        f"Three-Condition Comparison — {diary_id}\n"
        "(Baseline: raw diary text · Partial: emotion labels · Full: LLM-mediated visual design)",
        fontsize=11, y=1.02
    )

    for ax, cond in zip(axes, CONDITIONS):
        img_path = OUTPUTS / cond / f"{diary_id}.png"
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        clip_val = scores.get(diary_id, {}).get(cond, None)
        score_str = f"CLIP = {clip_val:.4f}" if clip_val is not None else ""
        label = LABELS[cond]
        color = COLORS[cond]

        ax.set_title(f"{label}\n{score_str}", fontsize=12, color=color, fontweight="bold", pad=8)

        # 高亮边框标出 Full（最优）
        if cond == "condition_full":
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    plt.tight_layout()
    out = OUTPUTS / "figure_condition_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存：{out}")


# ── Figure 2：10 × 3 CLIP 分热力图 ───────────────────────────────
def make_clip_heatmap(scores):
    diary_ids = sorted(scores.keys(), key=lambda x: int(x.split("_")[1]))
    data = np.array([
        [scores[d].get(c, np.nan) for c in CONDITIONS]
        for d in diary_ids
    ])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.18, vmax=0.31)

    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=11)
    ax.set_yticks(range(len(diary_ids)))
    ax.set_yticklabels([d.replace("diary_", "D") for d in diary_ids], fontsize=10)
    ax.set_title("CLIP Score Heatmap (10 Diaries × 3 Conditions)", fontsize=11, pad=10)

    # 在格子内写数值
    for i, d in enumerate(diary_ids):
        for j, c in enumerate(CONDITIONS):
            val = scores[d].get(c, np.nan)
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.265 else "white")

    plt.colorbar(im, ax=ax, label="CLIP Cosine Similarity", shrink=0.85)

    # 列均值标注（放在 x 轴下方）
    col_means = data.mean(axis=0)
    for j, (c, mean) in enumerate(zip(CONDITIONS, col_means)):
        ax.text(j, len(diary_ids) + 0.3, f"μ={mean:.4f}",
                ha="center", va="top", fontsize=8.5,
                color=COLORS[c], fontweight="bold")

    ax.set_ylim(len(diary_ids) + 0.8, -0.5)  # 留出均值行空间
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.12, right=0.88)
    out = OUTPUTS / "figure_clip_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存：{out}")


if __name__ == "__main__":
    scores = load_clip_scores()
    make_condition_comparison(scores, diary_id="diary_1")
    make_clip_heatmap(scores)
    print("\n完成。请检查 outputs/ 目录。")
