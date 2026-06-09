"""
生成报告配图，所有数据均来自实测结果（clip_scores.csv + 真实生成图像）。
产出：
  outputs/figure_pipeline.png           —— 水平系统架构图（顶刊风格）
  outputs/figure_qualitative_gallery.png —— 多日记三条件对比 gallery
  outputs/figure_clip_heatmap.png       —— 10篇日记 × 3条件 CLIP 分热力图
"""

import csv
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib
import numpy as np

# ── 中文字体（macOS）────────────────────────────────────────────────────────
for _font in ["STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).parent.parent
OUTPUTS  = BASE_DIR / "outputs"
CONDITIONS = ["condition_baseline", "condition_partial", "condition_full"]
LABELS     = {"condition_baseline": "Baseline",
              "condition_partial":  "Partial",
              "condition_full":     "Full"}
COLORS     = {"condition_baseline": "#2e75b6",
              "condition_partial":  "#c55a11",
              "condition_full":     "#375623"}

# ── 读取 CLIP 分 ─────────────────────────────────────────────────────────────
def load_clip_scores():
    scores = {}
    with open(OUTPUTS / "clip_scores.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d, c, s = row["diary"], row["condition"], float(row["clip_score"])
            scores.setdefault(d, {})[c] = s
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — 水平系统架构图
# ═══════════════════════════════════════════════════════════════════════════════
def make_pipeline_diagram():
    # ── 可调参数 ──────────────────────────────────────────────────────────────
    FW, FH = 16, 5.2          # 画布尺寸（英寸）
    CY     = 2.6              # 主流程中心 y
    MW, MH = 2.05, 1.05       # 主框 宽/高
    BW, BH = 2.0,  0.78       # 分支框 宽/高
    BY     = [3.85, 2.6, 1.35] # 三分支 y（上/中/下）

    # 六个节点 x 中心
    Xs = dict(input=1.35, llm=4.05, gate=6.75, branch=9.45, gen=12.15, eval=14.85)

    # ── 配色 ─────────────────────────────────────────────────────────────────
    C = {
        "input": ("#dce6f1", "#1f4e79"),
        "llm":   ("#dce6f1", "#2e75b6"),
        "gate":  ("#ede7f6", "#6a1b9a"),
        "base":  ("#dbeafe", "#1d4ed8"),
        "part":  ("#fef3c7", "#92400e"),
        "full":  ("#dcfce7", "#166534"),
        "gen":   ("#fce4d6", "#c55a11"),
        "eval":  ("#d9f0d3", "#375623"),
    }

    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── 绘制圆角矩形 ──────────────────────────────────────────────────────────
    def box(cx, cy, w, h, title, sub, fc, ec, fs=9.2, sfs=7.4):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.1", lw=1.6,
                           facecolor=fc, edgecolor=ec, zorder=3)
        ax.add_patch(r)
        dy = 0.17 if sub else 0
        ax.text(cx, cy + dy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=ec, zorder=4)
        if sub:
            ax.text(cx, cy - 0.2, sub, ha="center", va="center",
                    fontsize=sfs, color="#555", zorder=4)

    # ── 绘制箭头 ──────────────────────────────────────────────────────────────
    def arrow(x1, y1, x2, y2, col="#999"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.4, mutation_scale=13,
                                   shrinkA=5, shrinkB=5),
                    zorder=2)

    # ── 主框 ─────────────────────────────────────────────────────────────────
    box(Xs["input"],  CY, MW, MH,
        "Chinese Diary Input", "中文日记（非结构化文本）",
        *C["input"])

    box(Xs["llm"],    CY, MW, MH,
        "DeepSeek-Chat", "valence · arousal · metaphor · scene",
        *C["llm"])

    box(Xs["gate"],   CY, MW, 0.9,
        "Human Audit Gate", "emotion_audit.json",
        *C["gate"])

    box(Xs["gen"],    CY, MW, MH,
        "ChatGPT Images 2.0", "像素艺术生成  ·  30 张图",
        *C["gen"])

    box(Xs["eval"],   CY, MW, 0.9,
        "CLIP ViT-B/32", "cosine similarity 评分",
        *C["eval"])

    # ── 分支框 ───────────────────────────────────────────────────────────────
    branch_meta = [
        ("Baseline", "Raw diary text\n→ translate + pixel style", "base"),
        ("Partial",  "Emotion labels\n+ valence / arousal",        "part"),
        ("Full",     "Core metaphor + scene\n+ color + composition","full"),
    ]
    for by, (label, sub, key) in zip(BY, branch_meta):
        box(Xs["branch"], by, BW, BH, label, sub, *C[key], fs=9.0, sfs=7.2)

    # ── 箭头：主流程 ─────────────────────────────────────────────────────────
    arrow(Xs["input"] + MW/2,  CY, Xs["llm"]  - MW/2, CY)
    arrow(Xs["llm"]   + MW/2,  CY, Xs["gate"] - MW/2, CY)

    # Gate → 三分支（扇出）
    for by in BY:
        arrow(Xs["gate"] + MW/2, CY, Xs["branch"] - BW/2, by)

    # 三分支 → Gen（扇入）
    for by in BY:
        arrow(Xs["branch"] + BW/2, by, Xs["gen"] - MW/2, CY)

    arrow(Xs["gen"] + MW/2, CY, Xs["eval"] - MW/2, CY)

    # ── 标题 ─────────────────────────────────────────────────────────────────
    ax.text(FW/2, FH - 0.3,
            "System Architecture: NLP-Mediated Pixel Art Emotion Visualization",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color="#222")

    out = OUTPUTS / "figure_pipeline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ 保存：{out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — 多日记三条件 Qualitative Gallery
# ═══════════════════════════════════════════════════════════════════════════════
def make_qualitative_gallery(scores):
    # ── 展示的日记（均为非私人日记）───────────────────────────────────────────
    GALLERY = [
        {
            "id":      "diary_5",
            "emotion": "Anger  |  valence −0.7  arousal 0.8",
            "excerpt": "今天下午的面谈让我到现在心里都堵得慌。对方几乎没怎么抬头看我，"
                       "打断我的自我介绍，轻飘飘地说：跑个Demo也算熟悉AI吗？",
        },
        {
            "id":      "diary_7",
            "emotion": "Amusement  |  valence +0.1  arousal 0.7",
            "excerpt": "今晚和室友在实验室加班到十一点半，论文格式一团糟，代码也在报错。"
                       "窗外突然传来一声猫叫，听起来像班主任训人，两人笑到蹲地上。",
        },
        {
            "id":      "diary_9",
            "emotion": "Anxiety  |  valence −0.7  arousal 0.8",
            "excerpt": "刷朋友圈看到高中好友在上海新租公寓的照片，精巧落地窗、高耸写字楼。"
                       "她已利落迈入社会轨道，我却感到强烈的「被留下的恐惧」。",
        },
        {
            "id":      "diary_10",
            "emotion": "Triumph  |  valence +0.9  arousal 0.8",
            "excerpt": "终于！折磨了三天的API连接报错今晚解决了！"
                       "看到终端跳出绿色200 OK，我直接从椅子上蹦起来。",
        },
    ]

    # ── 画布参数 ──────────────────────────────────────────────────────────────
    NCOLS   = 4          # 信息列 + 3 图像列
    NROWS   = len(GALLERY)
    COL_W   = [2.8, 3.0, 3.0, 3.0]   # 每列宽度（英寸）
    ROW_H   = 3.2                      # 每行高度（英寸）
    HDR_H   = 0.55                     # 标题行高度

    FW = sum(COL_W) + 0.3
    FH = HDR_H + NROWS * ROW_H + 0.3

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    # 列边界（累加）
    col_x = [0.15]
    for w in COL_W:
        col_x.append(col_x[-1] + w)

    # ── 标题行 ────────────────────────────────────────────────────────────────
    hdr_y = FH - HDR_H + 0.05
    cond_labels = [("Baseline", COLORS["condition_baseline"]),
                   ("Partial",  COLORS["condition_partial"]),
                   ("Full",     COLORS["condition_full"])]
    for ci, (lbl, col) in enumerate(cond_labels):
        cx = (col_x[ci+1] + col_x[ci+2]) / 2
        fig.text(cx / FW, (hdr_y + HDR_H/2) / FH, lbl,
                 ha="center", va="center",
                 fontsize=11, fontweight="bold", color=col)

    # ── 每行日记 ─────────────────────────────────────────────────────────────
    for ri, entry in enumerate(GALLERY):
        did  = entry["id"]
        row_top = FH - HDR_H - ri * ROW_H
        row_bot = row_top - ROW_H
        row_cy  = (row_top + row_bot) / 2

        # 浅色行背景（隔行）
        if ri % 2 == 0:
            bg = plt.Rectangle((col_x[0]/FW, row_bot/FH),
                                (col_x[-1] - col_x[0])/FW, ROW_H/FH,
                                facecolor="#f8f8f8", edgecolor="none",
                                transform=fig.transFigure, zorder=0)
            fig.add_artist(bg)

        # ── 左列：情绪与摘要 ─────────────────────────────────────────────────
        cx_info = (col_x[0] + col_x[1]) / 2
        # 日记编号 + 情绪
        fig.text(cx_info/FW, (row_cy + 0.85)/FH,
                 did.replace("_", " ").upper(),
                 ha="center", va="center",
                 fontsize=9, fontweight="bold", color="#333")
        fig.text(cx_info/FW, (row_cy + 0.45)/FH,
                 entry["emotion"],
                 ha="center", va="center",
                 fontsize=7.5, color="#555")

        # 文字摘要（换行显示）
        fig.text(cx_info/FW, (row_cy - 0.2)/FH,
                 entry["excerpt"],
                 ha="center", va="center",
                 fontsize=6.8, color="#333",
                 wrap=True,
                 multialignment="center",
                 bbox=dict(boxstyle="round,pad=0.25",
                           facecolor="#eeeeee", edgecolor="#cccccc", lw=0.8))

        # CLIP 分数（小字）
        sc = scores.get(did, {})
        score_str = (f"B={sc.get('condition_baseline',0):.3f}  "
                     f"P={sc.get('condition_partial',0):.3f}  "
                     f"F={sc.get('condition_full',0):.3f}")
        # 高亮最高分
        best_val = max(sc.values()) if sc else 0
        fig.text(cx_info/FW, (row_cy - 1.1)/FH,
                 score_str,
                 ha="center", va="center",
                 fontsize=7, color="#333",
                 fontfamily="monospace")

        # ── 三列图像 ─────────────────────────────────────────────────────────
        for ci, cond in enumerate(CONDITIONS):
            img_path = OUTPUTS / cond / f"{did}.png"
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")

            # 图像区域（归一化坐标）
            pad = 0.12
            x0 = (col_x[ci+1] + pad) / FW
            y0 = (row_bot + pad) / FH
            w  = (COL_W[ci+1] - 2*pad) / FW
            h  = (ROW_H - pad - 0.55) / FH

            ax_img = fig.add_axes([x0, y0, w, h])
            ax_img.imshow(img)
            ax_img.axis("off")

            # 图像边框颜色高亮最优条件
            clip_val = sc.get(cond, 0)
            is_best  = (clip_val == best_val)
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(COLORS[cond] if is_best else "#cccccc")
                spine.set_linewidth(2.5 if is_best else 0.8)

            # 图像下方 CLIP 分
            cx_img = (col_x[ci+1] + col_x[ci+2]) / 2
            fig.text(cx_img/FW, (row_bot + 0.28)/FH,
                     f"CLIP = {clip_val:.4f}",
                     ha="center", va="center",
                     fontsize=7.5,
                     color=COLORS[cond] if is_best else "#777",
                     fontweight="bold" if is_best else "normal")

        # 行分隔线
        if ri < NROWS - 1:
            line = plt.Line2D([col_x[0]/FW, col_x[-1]/FW],
                              [row_bot/FH, row_bot/FH],
                              color="#dddddd", lw=0.8,
                              transform=fig.transFigure)
            fig.add_artist(line)

    # ── 外框 + 图注 ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "Bold border = highest CLIP score per diary. "
             "Diary excerpts translated to English for legibility; original Chinese used for generation.",
             ha="center", va="bottom", fontsize=7, color="#888")

    out = OUTPUTS / "figure_qualitative_gallery.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ 保存：{out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — CLIP 热力图
# ═══════════════════════════════════════════════════════════════════════════════
def make_clip_heatmap(scores):
    diary_ids = sorted(scores.keys(), key=lambda x: int(x.split("_")[1]))
    data = np.array([
        [scores[d].get(c, np.nan) for c in CONDITIONS]
        for d in diary_ids
    ])

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.18, vmax=0.31)

    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=11)
    ax.set_yticks(range(len(diary_ids)))
    ax.set_yticklabels([d.replace("diary_", "D") for d in diary_ids], fontsize=10)
    ax.set_title("CLIP Score Heatmap (10 Diaries × 3 Conditions)", fontsize=11, pad=10)

    for i, d in enumerate(diary_ids):
        for j, c in enumerate(CONDITIONS):
            val = scores[d].get(c, np.nan)
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.265 else "white")

    plt.colorbar(im, ax=ax, label="CLIP Cosine Similarity", shrink=0.85)

    col_means = data.mean(axis=0)
    for j, (c, mean) in enumerate(zip(CONDITIONS, col_means)):
        ax.text(j, len(diary_ids) + 0.3, f"μ={mean:.4f}",
                ha="center", va="top", fontsize=8.5,
                color=COLORS[c], fontweight="bold")

    ax.set_ylim(len(diary_ids) + 0.8, -0.5)
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.12, right=0.88)

    out = OUTPUTS / "figure_clip_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ 保存：{out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    scores = load_clip_scores()
    make_pipeline_diagram()
    make_qualitative_gallery(scores)
    make_clip_heatmap(scores)
    print("\n完成。请检查 outputs/ 目录。")
