"""
分析 Google Form 强迫选择结果。

输入：outputs/form_responses.csv（从 Google Forms → 回应 → 下载 .csv）
输出：win-rate 统计 + outputs/winrate_chart.png

CSV 格式（Google Forms 导出）：
  第 1 列：Timestamp
  第 2-11 列：10 道题的答案（各为 "A" / "B" / "C"）
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "sans-serif"]

BASE_DIR = Path(__file__).parent.parent
ANSWER_KEY_PATH = BASE_DIR / "outputs" / "answer_key.json"
DIARIES = [f"diary_{i}" for i in range(1, 11)]
CONDITIONS = ["full", "partial", "baseline"]


def load_answer_key():
    with open(ANSWER_KEY_PATH, encoding="utf-8") as f:
        return json.load(f)["shuffle"]


def load_responses():
    csv_path = BASE_DIR / "outputs" / "form_responses.csv"
    if not csv_path.exists():
        print("✗ 未找到 form_responses.csv")
        print("  请从 Google Forms → 回应 → 下载 .csv，保存到 outputs/form_responses.csv")
        return None
    df = pd.read_csv(csv_path)
    print(f"✓ 已读取 {len(df)} 份回应")
    return df


def decode_responses(df, shuffle):
    """将 A/B/C 答案映射回条件名称，返回长格式 DataFrame。"""
    # 跳过 Timestamp 列，取后 10 列
    answer_cols = df.columns[1:11]
    if len(answer_cols) < 10:
        raise ValueError(f"CSV 只找到 {len(answer_cols)} 道题列，预期 10 列（检查文件格式）")

    rows = []
    for rater_idx, row in df.iterrows():
        for q_idx, col in enumerate(answer_cols):
            diary_id = DIARIES[q_idx]
            letter = str(row[col]).strip().upper()
            if letter not in ("A", "B", "C"):
                continue
            condition = shuffle[diary_id][letter]
            rows.append({
                "rater": rater_idx,
                "diary": diary_id,
                "chosen_letter": letter,
                "chosen_condition": condition,
            })
    return pd.DataFrame(rows)


def compute_winrates(decoded):
    counts = defaultdict(int)
    totals = defaultdict(int)

    for _, row in decoded.iterrows():
        diary = row["diary"]
        chosen = row["chosen_condition"]
        for cond in CONDITIONS:
            totals[(diary, cond)] += 1
        counts[(diary, chosen)] += 1

    # 聚合：按条件统计总胜次 / 总比较次
    cond_wins  = {c: 0 for c in CONDITIONS}
    cond_total = {c: 0 for c in CONDITIONS}
    for (diary, cond), wins in counts.items():
        cond_wins[cond]  += wins
        cond_total[cond] += totals[(diary, cond)]

    print("\n" + "=" * 60)
    print("整体胜率（win rate across all diaries × raters）")
    print("=" * 60)
    for cond in CONDITIONS:
        n = cond_total[cond]
        w = cond_wins[cond]
        print(f"  {cond:10s}  {w:3d}/{n:3d}  = {w/n*100:.1f}%")

    print("\n" + "=" * 60)
    print("按日记的胜率（chosen condition per diary）")
    print("=" * 60)
    n_raters = decoded["rater"].nunique()
    print(f"{'日记':12s}  {'胜出条件':10s}  {'A/B/C 票数'}")
    for diary in DIARIES:
        sub = decoded[decoded["diary"] == diary]
        winner = sub["chosen_condition"].value_counts().idxmax()
        vote_str = "  ".join(
            f"{l}={int((sub['chosen_letter'] == l).sum())}"
            for l in ("A", "B", "C")
        )
        print(f"  {diary:12s}  {winner:10s}  {vote_str}")

    return cond_wins, cond_total


def plot_winrates(cond_wins, cond_total):
    labels = CONDITIONS
    rates  = [cond_wins[c] / cond_total[c] * 100 for c in labels]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rates, color=colors, width=0.5)
    ax.set_ylim(0, 100)
    ax.axhline(y=100/3, linestyle="--", color="gray", alpha=0.6, label="随机基线 (33.3%)")
    ax.set_ylabel("胜率 (%)")
    ax.set_title("强迫选择胜率（三条件消融实验）")
    ax.legend()
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11)

    out = BASE_DIR / "outputs" / "winrate_chart.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓ 图表已保存：{out}")


if __name__ == "__main__":
    shuffle = load_answer_key()
    df = load_responses()
    if df is None:
        raise SystemExit(1)

    decoded = decode_responses(df, shuffle)
    cond_wins, cond_total = compute_winrates(decoded)
    plot_winrates(cond_wins, cond_total)

    decoded_out = BASE_DIR / "outputs" / "decoded_responses.csv"
    decoded.to_csv(decoded_out, index=False)
    print(f"✓ 解码明细已保存：{decoded_out}")
