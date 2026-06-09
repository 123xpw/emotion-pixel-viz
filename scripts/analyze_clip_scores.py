"""
三条件 CLIP 得分统计分析
- Wilcoxon 符号秩检验（n=10，非参数，不假设正态分布）
- 效应量 r = Z / sqrt(N)
- 折线图 + 条形图
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]

BASE_DIR = Path(__file__).parent.parent
CONDITION_LABELS = {
    "condition_baseline": "Baseline",
    "condition_partial":  "Partial",
    "condition_full":     "Full",
}


def wilcoxon_with_effect(a, b, label):
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    n = len(a)
    # 效应量 r（rank-biserial correlation 近似）
    z = stats.norm.ppf(p / 2)
    r = abs(z) / np.sqrt(n)
    direction = ">" if a.mean() > b.mean() else "<"
    print(f"  {label}: W={stat:.1f}, p={p:.4f}{'*' if p<0.05 else ''}, r={r:.3f}  "
          f"({a.mean():.4f} {direction} {b.mean():.4f})")
    return p, r


def main():
    csv_path = BASE_DIR / "outputs" / "clip_scores.csv"
    df = pd.read_csv(csv_path)

    # 转宽表
    wide = df.pivot(index="diary", columns="condition", values="clip_score")
    bl = wide["condition_baseline"]
    pa = wide["condition_partial"]
    fu = wide["condition_full"]

    # ── 描述统计 ──
    print("=" * 55)
    print("各条件 CLIP 得分描述统计")
    print("=" * 55)
    for col, label in CONDITION_LABELS.items():
        s = wide[col]
        print(f"  {label:10s}  mean={s.mean():.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}")

    # ── 假设检验 ──
    print("\n" + "=" * 55)
    print("Wilcoxon 符号秩检验（双侧，n=10）")
    print("=" * 55)
    wilcoxon_with_effect(fu, bl, "Full   vs Baseline")
    wilcoxon_with_effect(fu, pa, "Full   vs Partial ")
    wilcoxon_with_effect(bl, pa, "Baseline vs Partial")

    # ── 逐日记得分宽表 ──
    print("\n" + "=" * 55)
    print("逐日记 CLIP 得分")
    print("=" * 55)
    display = wide.rename(columns=CONDITION_LABELS)
    display.index = [d.replace("diary_", "D") for d in display.index]
    display["Best"] = display.idxmax(axis=1)
    print(display.round(4).to_string())
    wins = display["Best"].value_counts()
    print(f"\n各条件「最高分」次数：{wins.to_dict()}")

    # ── 图1：条形图（均值 ± std）──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(CONDITION_LABELS.values())
    means  = [wide[c].mean() for c in CONDITION_LABELS]
    stds   = [wide[c].std()  for c in CONDITION_LABELS]
    colors = ["#90CAF9", "#FFCC80", "#A5D6A7"]

    bars = axes[0].bar(labels, means, yerr=stds, capsize=8,
                       color=colors, edgecolor="gray", width=0.5)
    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, m + 0.003,
                     f"{m:.4f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0.18, 0.30)
    axes[0].set_ylabel("CLIP Cosine Similarity")
    axes[0].set_title("Mean CLIP Score by Condition (±SD)")
    axes[0].grid(axis="y", alpha=0.3)

    # ── 图2：折线图（每条日记三条件得分）──
    x = np.arange(len(CONDITION_LABELS))
    diary_ids = [d.replace("diary_", "D") for d in wide.index]
    cmap = plt.cm.get_cmap("tab10", len(diary_ids))
    for i, did in enumerate(diary_ids):
        row = [wide.iloc[i][c] for c in CONDITION_LABELS]
        axes[1].plot(labels, row, marker="o", color=cmap(i),
                     linewidth=1.2, markersize=5, label=did)
    axes[1].set_ylabel("CLIP Cosine Similarity")
    axes[1].set_title("Per-Diary CLIP Scores Across Conditions")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = BASE_DIR / "outputs" / "clip_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存：outputs/clip_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
