"""
生成报告配图。
产出：
  outputs/figure_pipeline.svg / .png   —— 带数据样例的水平系统架构图
  outputs/figure_qualitative_gallery.png —— 多日记三条件 gallery
  outputs/figure_clip_heatmap.svg / .png —— CLIP 热力图
"""

import csv
from pathlib import Path
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── 中文字体（macOS）────────────────────────────────────────────────
for _font in ["STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR   = Path(__file__).parent.parent
OUTPUTS    = BASE_DIR / "outputs"
CONDITIONS = ["condition_baseline", "condition_partial", "condition_full"]
LABELS     = {"condition_baseline": "Baseline",
              "condition_partial":  "Partial",
              "condition_full":     "Full"}
C_COLOR    = {"condition_baseline": "#1d4ed8",
              "condition_partial":  "#92400e",
              "condition_full":     "#166534"}

# ── 读取 CLIP 分 ─────────────────────────────────────────────────────
def load_clip_scores():
    scores = {}
    with open(OUTPUTS / "clip_scores.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d, c, s = row["diary"], row["condition"], float(row["clip_score"])
            scores.setdefault(d, {})[c] = s
    return scores


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — 带数据样例的系统架构图（SVG 可编辑输出）
# ═══════════════════════════════════════════════════════════════════
#
# 布局：上方流程条（方框+箭头） + 下方数据样例行
# 可调参数集中在 LAYOUT 字典，修改一处即可重排
#
LAYOUT = dict(
    figw   = 17,     # 画布宽（英寸）
    figh   = 8.5,    # 画布高（英寸）
    flow_y = 6.8,    # 上方流程框中心 y
    ex_y   = 3.2,    # 下方样例区中心 y（大约）
    mw     = 2.1,    # 主框宽
    mh     = 1.1,    # 主框高
    bw     = 1.9,    # 分支框宽
    bh     = 0.72,   # 分支框高
    branch_ys = [7.85, 6.8, 5.75],  # 三分支 y（上/中/下）
)

# 六节点 x 中心
NODE_X = dict(input=1.3, llm=4.1, gate=6.9, branch=9.6, gen=12.4, eval=15.2)

# 节点配色 (face, edge)
NODE_C = dict(
    input  = ("#e8f0fe", "#1a56db"),
    llm    = ("#e8f0fe", "#1a56db"),
    gate   = ("#f3e8ff", "#7e22ce"),
    base   = ("#dbeafe", "#1d4ed8"),
    part   = ("#fff7ed", "#c2410c"),
    full   = ("#dcfce7", "#15803d"),
    gen    = ("#fff1f0", "#c0392b"),
    eval_  = ("#f0fdf4", "#166534"),
)


def _box(ax, cx, cy, w, h, title, subtitle, fc, ec, tfs=9, sfs=7.2):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.12", lw=1.8,
                       facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(r)
    dy = 0.19 if subtitle else 0
    ax.text(cx, cy + dy, title,
            ha="center", va="center", fontsize=tfs,
            fontweight="bold", color=ec, zorder=4)
    if subtitle:
        ax.text(cx, cy - 0.21, subtitle,
                ha="center", va="center", fontsize=sfs,
                color="#555", zorder=4)


def _arr(ax, x1, y1, x2, y2, col="#999"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.5, mutation_scale=13,
                                shrinkA=5, shrinkB=5), zorder=2)


def _sample_box(ax, cx, cy, w, h, lines, fc="#f9f9f9", ec="#cccccc"):
    """在样例区绘制带内容的小矩形（展示真实数据）。"""
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.1", lw=1.0,
                       facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(r)
    n = len(lines)
    for i, (txt, fs, col, bold) in enumerate(lines):
        offset = (n / 2 - i - 0.5) * (h / (n + 0.5))
        ax.text(cx, cy + offset, txt,
                ha="center", va="center", fontsize=fs, color=col,
                fontweight="bold" if bold else "normal",
                zorder=4, wrap=False)


def make_pipeline_diagram():
    L  = LAYOUT
    NX = NODE_X

    fig, ax = plt.subplots(figsize=(L["figw"], L["figh"]))
    ax.set_xlim(0, L["figw"])
    ax.set_ylim(0, L["figh"])
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fy = L["flow_y"]
    mw, mh = L["mw"], L["mh"]
    bw, bh = L["bw"], L["bh"]
    bys    = L["branch_ys"]

    # ── 上方：流程框 ──────────────────────────────────────────────────
    _box(ax, NX["input"],  fy, mw, mh,
         "Chinese Diary", "中文日记输入",
         *NODE_C["input"])

    _box(ax, NX["llm"],    fy, mw, mh,
         "DeepSeek-Chat", "情绪结构化提取",
         *NODE_C["llm"])

    _box(ax, NX["gate"],   fy, mw, 0.9,
         "Human Audit Gate", "emotion_audit.json",
         *NODE_C["gate"])

    for by, (lbl, sub, key) in zip(bys, [
        ("Baseline", "原始译文 + 风格词", "base"),
        ("Partial",  "情绪标签 + v/a",    "part"),
        ("Full",     "隐喻 + 场景 + 色调", "full"),
    ]):
        _box(ax, NX["branch"], by, bw, bh, lbl, sub,
             *NODE_C[key], tfs=8.5, sfs=7)

    _box(ax, NX["gen"],    fy, mw, mh,
         "ChatGPT Images 2.0", "像素艺术生成 × 30",
         *NODE_C["gen"])

    _box(ax, NX["eval"],   fy, mw, 0.9,
         "CLIP ViT-B/32", "余弦相似度评分",
         *NODE_C["eval_"])

    # ── 上方：箭头 ───────────────────────────────────────────────────
    _arr(ax, NX["input"] + mw/2, fy, NX["llm"]  - mw/2, fy)
    _arr(ax, NX["llm"]  + mw/2, fy, NX["gate"]  - mw/2, fy)
    for by in bys:
        _arr(ax, NX["gate"]   + mw/2, fy,   NX["branch"] - bw/2, by)
        _arr(ax, NX["branch"] + bw/2, by,   NX["gen"]    - mw/2, fy)
    _arr(ax, NX["gen"]  + mw/2, fy, NX["eval"]  - mw/2, fy)

    # ── 下方：数据样例区（浅灰底板）─────────────────────────────────
    bg = FancyBboxPatch((0.2, 0.25), L["figw"] - 0.4, 4.5,
                        boxstyle="round,pad=0.1", lw=1.0,
                        facecolor="#f7f7f7", edgecolor="#dddddd", zorder=1)
    ax.add_patch(bg)
    ax.text(0.55, 4.62, "Example Data Trace  (diary_5: anger)",
            fontsize=8.5, color="#888", style="italic", va="bottom")

    # 样例1：日记输入
    _sample_box(ax, NX["input"], 2.8, mw + 0.6, 3.5, [
        ("INPUT DIARY",      7,   "#1a56db", True),
        ("今天下午的面谈",    7.8, "#222",    False),
        ("让我心里都堵得慌。", 7.8, "#222",    False),
        ("HR 打断我说：",     7.8, "#222",    False),
        ("\"跑个 Demo 也",   7.8, "#555",    False),
        ("算熟悉AI吗？\"",   7.8, "#c0392b", False),
    ], fc="#eef4ff", ec="#1a56db")

    # 连接线（上方框 → 样例框）
    ax.plot([NX["input"], NX["input"]], [fy - mh/2, 4.55],
            color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # 样例2：DeepSeek 输出 JSON
    _sample_box(ax, NX["llm"], 2.8, mw + 0.6, 3.5, [
        ("EMOTION JSON",      7,   "#1a56db", True),
        ("primary: anger",    7.5, "#166534", False),
        ("valence:  -0.7",    7.5, "#c0392b", False),
        ("arousal:   0.8",    7.5, "#c0392b", False),
        ("metaphor:",         7.5, "#222",    False),
        ("\"羞耻感，无处可藏\"", 7,  "#555",    False),
    ], fc="#eef4ff", ec="#1a56db")
    ax.plot([NX["llm"], NX["llm"]], [fy - mh/2, 4.55],
            color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # 样例3：三条件 Prompt 摘要
    ex_px = NX["branch"]
    _sample_box(ax, ex_px, 3.5, bw + 0.5, 2.2, [
        ("PROMPTS",                    7,   "#444",    True),
        ("B: raw diary → translate",   7.2, "#1d4ed8", False),
        ("P: anger, low v, high a",    7.2, "#c2410c", False),
        ("F: 羞耻感，隐喻驱动",        7.2, "#15803d", False),
        ("   + 灼热红 / 暗红色调",     7.0, "#15803d", False),
    ], fc="#fffbf0", ec="#999")
    ax.plot([ex_px, ex_px], [min(bys) - bh/2, 2.62],
            color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # 样例4：生成图像缩略图
    img_cx = NX["gen"]
    thumb_y0, thumb_h = 0.45, 3.8
    thumb_w_total = mw + 0.5
    iw = thumb_w_total / 3 - 0.08
    for ci, cond in enumerate(CONDITIONS):
        img_path = OUTPUTS / cond / "diary_5.png"
        if not img_path.exists():
            continue
        img   = Image.open(img_path).convert("RGB")
        xi    = img_cx - thumb_w_total/2 + ci * (iw + 0.08) + iw/2
        xi0   = xi - iw/2
        yi0   = thumb_y0
        ax_i  = ax.inset_axes(
            [xi0/L["figw"], yi0/L["figh"], iw/L["figw"], (thumb_h - 0.5)/L["figh"]])
        ax_i.imshow(img)
        ax_i.axis("off")
        col = list(C_COLOR.values())[ci]
        for sp in ax_i.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(1.4)
        ax.text(xi, thumb_y0 + thumb_h - 0.35,
                LABELS[cond], ha="center", va="center",
                fontsize=7, color=col, fontweight="bold")
    ax.plot([img_cx, img_cx], [fy - mh/2, 4.55],
            color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # 样例5：CLIP 分
    sc = load_clip_scores().get("diary_5", {})
    _sample_box(ax, NX["eval"], 2.8, mw + 0.3, 3.5, [
        ("CLIP SCORES",                            7,   "#166534", True),
        (f"Baseline  {sc.get('condition_baseline',0):.4f}", 7.5, "#1d4ed8", False),
        (f"Partial    {sc.get('condition_partial',0):.4f}", 7.5, "#c2410c", False),
        (f"Full         {sc.get('condition_full',0):.4f}",  7.5, "#15803d", True),
        ("↑ Full wins",                            7.5, "#15803d", True),
    ], fc="#f0fdf4", ec="#166534")
    ax.plot([NX["eval"], NX["eval"]], [fy - mh/2, 4.55],
            color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # ── 标题 & 注释 ──────────────────────────────────────────────────
    ax.text(L["figw"]/2, L["figh"] - 0.22,
            "System Architecture — NLP-Mediated Pixel Art Emotion Visualization",
            ha="center", va="top", fontsize=12, fontweight="bold", color="#111")
    ax.text(0.3, 0.12,
            "Dashed lines connect pipeline stages to their corresponding data examples (diary_5, anger).",
            ha="left", va="bottom", fontsize=7.5, color="#888")

    # ── 保存 SVG + PNG ───────────────────────────────────────────────
    for ext in ("svg", "png"):
        out = OUTPUTS / f"figure_pipeline.{ext}"
        dpi = 200 if ext == "png" else None
        kw  = dict(bbox_inches="tight", facecolor="white")
        if dpi:
            kw["dpi"] = dpi
        plt.savefig(out, format=ext, **kw)
        print(f"✓ 保存：{out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — 多日记 Qualitative Gallery（高分辨率 PNG）
# ═══════════════════════════════════════════════════════════════════
GALLERY_DIARIES = [
    {"id": "diary_5",  "emotion": "Anger",     "v": -0.7, "a": 0.8,
     "excerpt": "HR打断自我介绍说：\n\"跑个Demo也算熟悉AI吗？\"\n脸上一阵滚烫，愤怒无处发泄。"},
    {"id": "diary_7",  "emotion": "Amusement", "v":  0.1, "a": 0.7,
     "excerpt": "论文格式一团糟，代码在报错。\n窗外传来猫叫像班主任训人，\n两人笑到蹲在地上。"},
    {"id": "diary_9",  "emotion": "Anxiety",   "v": -0.7, "a": 0.8,
     "excerpt": "好友在上海新租精巧公寓，\n已利落迈入社会轨道。\n我感到「被留下的恐惧」。"},
    {"id": "diary_10", "emotion": "Triumph",   "v":  0.9, "a": 0.8,
     "excerpt": "三天的API报错今晚解决！\n终端出现绿色200 OK，\n我从椅子上蹦了起来。"},
]


def make_qualitative_gallery(scores):
    COL_INFO = 3.0
    COL_IMG  = 2.9
    ROW_H    = 3.1
    HDR_H    = 0.5
    PAD      = 0.15
    NROWS    = len(GALLERY_DIARIES)

    FW = COL_INFO + 3 * COL_IMG + 0.3
    FH = HDR_H + NROWS * ROW_H + 0.2

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    col_x = [PAD, PAD + COL_INFO,
             PAD + COL_INFO + COL_IMG,
             PAD + COL_INFO + 2*COL_IMG,
             PAD + COL_INFO + 3*COL_IMG]

    # ── 列标题 ────────────────────────────────────────────────────────
    for ci, cond in enumerate(CONDITIONS):
        cx = (col_x[ci+1] + col_x[ci+2]) / 2
        fig.text(cx/FW, (FH - HDR_H/2)/FH,
                 LABELS[cond],
                 ha="center", va="center",
                 fontsize=11, fontweight="bold",
                 color=C_COLOR[cond])

    # ── 每行 ──────────────────────────────────────────────────────────
    for ri, entry in enumerate(GALLERY_DIARIES):
        did   = entry["id"]
        sc    = scores.get(did, {})
        top_y = FH - HDR_H - ri * ROW_H
        bot_y = top_y - ROW_H
        mid_y = (top_y + bot_y) / 2

        # 隔行底色
        if ri % 2 == 0:
            fig.add_artist(plt.Rectangle(
                (col_x[0]/FW, bot_y/FH),
                (col_x[-1] - col_x[0])/FW, ROW_H/FH,
                facecolor="#f8f9fa", edgecolor="none",
                transform=fig.transFigure, zorder=0))

        # 左列信息
        cx_i = (col_x[0] + col_x[1]) / 2
        fig.text(cx_i/FW, (mid_y + 0.9)/FH,
                 f"{did.upper().replace('_',' ')}",
                 ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color="#222")
        fig.text(cx_i/FW, (mid_y + 0.52)/FH,
                 f"{entry['emotion']}  ·  v={entry['v']:+.1f}  a={entry['a']:.1f}",
                 ha="center", va="center",
                 fontsize=7.8, color="#555")
        fig.text(cx_i/FW, (mid_y - 0.15)/FH,
                 entry["excerpt"],
                 ha="center", va="center",
                 fontsize=7, color="#333",
                 multialignment="center",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#eeeeee",
                           edgecolor="#cccccc", lw=0.8))

        # CLIP 分数行
        best_val = max(sc.values()) if sc else 0
        score_parts = []
        for cond in CONDITIONS:
            v   = sc.get(cond, 0)
            lbl = LABELS[cond][0]   # B / P / F
            score_parts.append(f"{lbl}={v:.3f}")
        fig.text(cx_i/FW, (bot_y + 0.28)/FH,
                 "  ".join(score_parts),
                 ha="center", va="center",
                 fontsize=7.2, color="#444",
                 fontfamily="monospace")

        # 三张图像
        for ci, cond in enumerate(CONDITIONS):
            img_path = OUTPUTS / cond / f"{did}.png"
            if not img_path.exists():
                continue
            img  = Image.open(img_path).convert("RGB")
            clip = sc.get(cond, 0)
            best = (clip == best_val)

            x0 = (col_x[ci+1] + 0.12) / FW
            y0 = (bot_y + 0.5) / FH
            w  = (COL_IMG - 0.24) / FW
            h  = (ROW_H - 0.75) / FH

            ax_i = fig.add_axes([x0, y0, w, h])
            ax_i.imshow(img)
            ax_i.axis("off")
            col = C_COLOR[cond]
            for sp in ax_i.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(col if best else "#cccccc")
                sp.set_linewidth(2.5 if best else 0.8)

            cx_img = (col_x[ci+1] + col_x[ci+2]) / 2
            fig.text(cx_img/FW, (bot_y + 0.28)/FH,
                     f"CLIP = {clip:.4f}" + (" ★" if best else ""),
                     ha="center", va="center",
                     fontsize=7.2,
                     color=col if best else "#888",
                     fontweight="bold" if best else "normal")

        # 行分隔线
        if ri < NROWS - 1:
            fig.add_artist(plt.Line2D(
                [col_x[0]/FW, col_x[-1]/FW],
                [bot_y/FH, bot_y/FH],
                color="#dddddd", lw=0.8,
                transform=fig.transFigure))

    # 底部注释
    fig.text(0.5, 0.005,
             "★ = highest CLIP score per diary row.  "
             "Bold border indicates condition with best image-text alignment.",
             ha="center", va="bottom", fontsize=7, color="#999")

    out = OUTPUTS / "figure_qualitative_gallery.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ 保存：{out}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — CLIP 热力图（SVG + PNG 双输出）
# ═══════════════════════════════════════════════════════════════════
def make_clip_heatmap(scores):
    diary_ids = sorted(scores.keys(), key=lambda x: int(x.split("_")[1]))
    data = np.array([[scores[d].get(c, np.nan) for c in CONDITIONS]
                     for d in diary_ids])

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
                        fontsize=8,
                        color="black" if val < 0.265 else "white")

    plt.colorbar(im, ax=ax, label="CLIP Cosine Similarity", shrink=0.85)
    col_means = data.mean(axis=0)
    for j, (c, m) in enumerate(zip(CONDITIONS, col_means)):
        ax.text(j, len(diary_ids) + 0.3, f"μ={m:.4f}",
                ha="center", va="top", fontsize=8.5,
                color=list(C_COLOR.values())[j], fontweight="bold")
    ax.set_ylim(len(diary_ids) + 0.8, -0.5)
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.12, right=0.88)

    for ext in ("svg", "png"):
        out = OUTPUTS / f"figure_clip_heatmap.{ext}"
        kw  = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kw["dpi"] = 150
        plt.savefig(out, format=ext, **kw)
        print(f"✓ 保存：{out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    scores = load_clip_scores()
    make_pipeline_diagram()
    make_qualitative_gallery(scores)
    make_clip_heatmap(scores)
    print("\n完成。SVG 文件可用 Illustrator / Inkscape 直接编辑。")
