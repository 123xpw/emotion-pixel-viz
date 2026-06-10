"""
generate_report_figures.py  —  Top-journal quality figure generation
Outputs:
  figure_pipeline.svg/.png          Pure architecture (Phase Band, no data trace)
  figure_qualitative_gallery.png    Streamlined 4×3 gallery (image-first)
  figure_clip_heatmap.svg/.png      Heatmap with Mean row + D9 annotation
  figure_clip_delta.svg/.png        NEW: ΔCLIP per diary (ablation effect)
"""

import csv
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# ─── Font setup ───────────────────────────────────────────────────────────────
# Prefer professional sans-serif; fall back to DejaVu Sans (always available).
def _best_font(candidates, fallback="DejaVu Sans"):
    avail = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in avail:
            return name
    return fallback

SANS = _best_font(["Arial", "Helvetica Neue", "Helvetica", "Liberation Sans"])

matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    [SANS, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ─── Condition palette (ColorBrewer RdYlBu, print + colorblind safe) ─────────
C_BL = "#2166AC"   # Baseline — blue
C_PT = "#D73027"   # Partial  — red
C_FL = "#1A9850"   # Full     — green

CONDITIONS = ["condition_baseline", "condition_partial", "condition_full"]
LABELS     = {"condition_baseline": "Baseline",
               "condition_partial":  "Partial",
               "condition_full":     "Full"}
C_COLOR    = {"condition_baseline": C_BL,
               "condition_partial":  C_PT,
               "condition_full":     C_FL}

BASE_DIR = Path(__file__).parent.parent
OUTPUTS  = BASE_DIR / "outputs"


# ─── Data loader ─────────────────────────────────────────────────────────────
def load_clip_scores():
    scores = {}
    with open(OUTPUTS / "clip_scores.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d, c, s = row["diary"], row["condition"], float(row["clip_score"])
            scores.setdefault(d, {})[c] = s
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — System Architecture (pure Phase Band, no inline data trace)
# ═══════════════════════════════════════════════════════════════════════════════
def make_pipeline_diagram():
    FW, FH = 8.5, 8.4          # double-column A4 width

    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")

    XLEFT, XRIGHT = 0.18, 8.32
    BW      = XRIGHT - XLEFT
    BADGE_X = XLEFT + 0.36

    # (y_bot, y_top, bg, border, badge_fill, num, label)
    BANDS = [
        (6.7, 7.7,   "#EEF3FF", "#3B82F6", "#2563EB", "1", "Input"),
        (4.65, 6.35, "#F5F0FF", "#7C3AED", "#7C3AED", "2", "Emotion Extraction"),
        (2.15, 4.30, "#FFFBEB", "#F59E0B", "#D97706", "3", "Prompt Construction"),
        (0.93, 1.80, "#F0FDF4", "#16A34A", "#16A34A", "4", "Image Generation"),
        (0.05, 0.82, "#ECFDF5", "#059669", "#059669", "5", "Evaluation"),
    ]

    for yb, yt, bgc, ec, bfc, num, lbl in BANDS:
        # drop shadow
        ax.add_patch(FancyBboxPatch(
            (XLEFT + 0.07, yb - 0.07), BW, yt - yb,
            boxstyle="round,pad=0.08", lw=0, facecolor="#00000012", zorder=1))
        # band background
        ax.add_patch(FancyBboxPatch(
            (XLEFT, yb), BW, yt - yb,
            boxstyle="round,pad=0.08", lw=1.5, facecolor=bgc, edgecolor=ec, zorder=2))
        mid = (yb + yt) / 2
        # step badge
        ax.add_patch(plt.Circle((BADGE_X, mid), 0.22, facecolor=bfc, zorder=7))
        ax.text(BADGE_X, mid, num, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=8)
        # phase label (top-left of band, right of badge)
        ax.text(XLEFT + 0.74, yt - 0.14, lbl, ha="left", va="top",
                fontsize=9, fontweight="bold", color=ec, zorder=7)

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def mbox(cx, cy, w, h, title, sub=None,
             fc="white", ec="#999", tc="#111", sc="#555",
             tfs=9.5, sfs=8.0):
        """Rounded module box with title + optional subtitle."""
        ax.add_patch(FancyBboxPatch(
            (cx - w/2 + 0.05, cy - h/2 - 0.05), w, h,
            boxstyle="round,pad=0.07", lw=0, facecolor="#00000015", zorder=4))
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.07", lw=1.8, facecolor=fc, edgecolor=ec, zorder=5))
        nlines = sub.count("\n") + 1 if sub else 0
        t_dy = 0.16 if sub else 0
        ax.text(cx, cy + t_dy, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=tc, zorder=6)
        if sub:
            ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                    fontsize=sfs, color=sc, zorder=6, linespacing=1.4)

    def arr(x1, y1, x2, y2, col="#AAAAAA", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                            mutation_scale=13, shrinkA=4, shrinkB=4), zorder=6)

    def hline(x1, x2, y, col="#CCCCCC", lw=1.5):
        ax.plot([x1, x2], [y, y], color=col, lw=lw, zorder=3)

    def vline(x, y1, y2, col="#CCCCCC", lw=1.5):
        ax.plot([x, x], [y1, y2], color=col, lw=lw, zorder=3)

    # ── Band 1: Input ─────────────────────────────────────────────────────────
    B1Y = (6.7 + 7.7) / 2   # 7.2
    mbox(4.25, B1Y, 6.3, 0.78,
         "Chinese Diary Input",
         "10 diary entries — personal emotion logs (Mandarin)",
         fc="#DBEAFE", ec="#2563EB", tc="#1E40AF", sc="#3B82F6")

    # Band 1 → Band 2  (main pipeline)
    arr(4.25, 6.70, 4.25, 6.35)

    # Baseline bypass: from Band 1 left edge → left gutter → Baseline box top
    BL_X = 1.88   # Baseline box centre-x (same as COND_XS[0] defined below)
    bypass_x = XLEFT + 0.04   # left gutter ~0.22
    ax.annotate("", xy=(BL_X, 4.30), xytext=(bypass_x, 4.30),
        arrowprops=dict(arrowstyle="-|>", color=C_BL, lw=1.8,
                        mutation_scale=12, shrinkA=0, shrinkB=4), zorder=6)
    ax.plot([bypass_x, bypass_x], [6.70, 4.30], color=C_BL, lw=1.8, zorder=5)
    ax.plot([XLEFT + 0.12, bypass_x], [6.70, 6.70], color=C_BL, lw=1.8, zorder=5)
    ax.text(bypass_x - 0.02, (6.70 + 4.30) / 2,
            "bypass\n(No LLM)", ha="right", va="center",
            fontsize=6.5, color=C_BL, style="italic", zorder=7)

    # ── Band 2: Emotion Extraction ────────────────────────────────────────────
    B2Y = (4.65 + 6.35) / 2   # 5.5
    mbox(2.85, B2Y, 2.9, 1.0,
         "DeepSeek-Chat",
         "Structured emotion extraction\n(JSON: emotion, metaphor, palette)",
         fc="#DDD6FE", ec="#7C3AED", tc="#5B21B6", sc="#7C3AED")
    arr(4.30, B2Y, 5.05, B2Y, col="#7C3AED")
    mbox(6.30, B2Y, 2.75, 1.0,
         "Human Audit Gate",
         "Manual verification of\nemotion JSON per diary",
         fc="#EDE9FE", ec="#9333EA", tc="#6B21A8", sc="#9333EA")

    # Band 2 → Band 3  fan-out from Audit Gate — only to Partial and Full
    COND_XS = [1.88, 4.25, 6.62]
    FAN_Y   = (4.65 + 4.30) / 2   # 4.475, in the inter-band gap

    vline(6.30, 4.65, FAN_Y)
    # fan-out bar only spans Partial → Full (skip Baseline)
    hline(COND_XS[1], 6.30, FAN_Y)
    for cx in COND_XS[1:]:   # Partial and Full only
        arr(cx, FAN_Y, cx, 4.30)

    # ── Band 3: Prompt Construction ───────────────────────────────────────────
    BOX_Y = 3.55          # condition box centres (upper half of Band 3)
    BOX_H = 1.1
    BOX_W = 2.05

    COND_CFG = [
        ("Baseline",
         "Raw auto-translation\n+ pixel art style token",
         "#C7D9FF", "#1D4ED8", "#1E40AF", "#2563EB"),
        ("Partial",
         "Emotion labels: primary,\nvalence, arousal + style",
         "#FFE4CC", "#EA580C", "#9A3412", "#C2410C"),
        ("Full",
         "Core metaphor + scene\n+ palette + no-text rule",
         "#C6F5D8", "#15803D", "#14532D", "#16A34A"),
    ]
    for cx, (t, s, fc, ec, tc, sc) in zip(COND_XS, COND_CFG):
        mbox(cx, BOX_Y, BOX_W, BOX_H, t, s, fc=fc, ec=ec, tc=tc, sc=sc)

    # Band 3 → Band 4  fan-in
    COLL_Y = 2.38   # collect bar inside lower Band 3
    for cx in COND_XS:
        vline(cx, BOX_Y - BOX_H / 2, COLL_Y)
    hline(COND_XS[0], COND_XS[-1], COLL_Y)
    arr(4.25, COLL_Y, 4.25, 1.80)

    # ── Band 4: Image Generation ──────────────────────────────────────────────
    B4Y = (0.93 + 1.80) / 2   # 1.365
    mbox(4.25, B4Y, 6.5, 0.62,
         "ChatGPT Images 2.0  (gpt-image-2)",
         "30 pixel art images  (10 diaries × 3 conditions)",
         fc="#DCFCE7", ec="#16A34A", tc="#14532D", sc="#15803D")

    # Band 4 → Band 5
    arr(4.25, 0.93, 4.25, 0.82)

    # ── Band 5: Evaluation ────────────────────────────────────────────────────
    B5Y = (0.05 + 0.82) / 2   # 0.435
    mbox(4.25, B5Y, 6.5, 0.57,
         "CLIP ViT-B/32 — cosine similarity (image ↔ emotion reference text)",
         "CLIP Score comparison across 3 conditions",
         fc="#D1FAE5", ec="#059669", tc="#064E3B", sc="#047857",
         tfs=8.5, sfs=7.5)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(FW / 2, FH - 0.07,
            "System Architecture — NLP-Mediated Pixel Art Emotion Visualization",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color="#111")

    for ext in ("svg", "png"):
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kw["dpi"] = 300
        plt.savefig(OUTPUTS / f"figure_pipeline.{ext}", format=ext, **kw)
        print(f"✓ figure_pipeline.{ext}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Qualitative Gallery (streamlined, image-first layout)
# ═══════════════════════════════════════════════════════════════════════════════
GALLERY_DIARIES = [
    {"id": "diary_5",  "emotion": "Anger",     "v": -0.7, "a": 0.8},
    {"id": "diary_7",  "emotion": "Amusement", "v":  0.1, "a": 0.7},
    {"id": "diary_9",  "emotion": "Anxiety",   "v": -0.7, "a": 0.8},
    {"id": "diary_10", "emotion": "Triumph",   "v":  0.9, "a": 0.8},
]


def make_qualitative_gallery(scores):
    NROWS  = len(GALLERY_DIARIES)
    INFO_W = 1.45   # left info column width (inches)
    IMG_W  = 2.55   # each image column width (inches)
    ROW_H  = 2.70   # height per row (image + score label)
    GAP_V  = 0.14   # vertical gap between rows
    HDR_H  = 0.50   # column header row
    MARG_B = 0.22   # bottom margin for footnote
    MARG_R = 0.12

    FW = INFO_W + 3 * IMG_W + MARG_R
    FH = HDR_H + NROWS * (ROW_H + GAP_V) + MARG_B

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    # ── Column headers ────────────────────────────────────────────────────────
    for ci, cond in enumerate(CONDITIONS):
        x_ctr = (INFO_W + (ci + 0.5) * IMG_W) / FW
        fig.text(x_ctr, (FH - HDR_H * 0.42) / FH,
                 LABELS[cond], ha="center", va="center",
                 fontsize=12, fontweight="bold", color=C_COLOR[cond])

    # ── Rows ─────────────────────────────────────────────────────────────────
    for ri, entry in enumerate(GALLERY_DIARIES):
        did = entry["id"]
        sc  = scores.get(did, {})

        row_top = FH - HDR_H - ri * (ROW_H + GAP_V)
        row_bot = row_top - ROW_H
        mid_y   = (row_top + row_bot) / 2

        # Alternating row tint
        if ri % 2 == 0:
            fig.add_artist(plt.Rectangle(
                (0, row_bot / FH), 1.0, ROW_H / FH,
                facecolor="#F7F8FA", edgecolor="none",
                transform=fig.transFigure, zorder=0))

        # Left info column
        icx = (INFO_W / 2) / FW
        fig.text(icx, (mid_y + 0.55) / FH,
                 did.replace("_", " ").title(),
                 ha="center", va="center",
                 fontsize=10.5, fontweight="bold", color="#111")
        fig.text(icx, (mid_y + 0.16) / FH,
                 entry["emotion"],
                 ha="center", va="center",
                 fontsize=9.5, color="#333", style="italic")
        fig.text(icx, (mid_y - 0.20) / FH,
                 f"v = {entry['v']:+.1f},  a = {entry['a']:.1f}",
                 ha="center", va="center",
                 fontsize=8.5, color="#666")

        # Three images
        best_val = max(sc.values()) if sc else 0
        IMG_H    = ROW_H - 0.52     # vertical space for image (leaves room for score)
        IMG_W_IN = IMG_W  - 0.22    # image width minus inner margin

        for ci, cond in enumerate(CONDITIONS):
            img_path = OUTPUTS / cond / f"{did}.png"
            if not img_path.exists():
                continue
            img  = Image.open(img_path).convert("RGB")
            clip = sc.get(cond, 0)
            best = abs(clip - best_val) < 1e-9
            col  = C_COLOR[cond]

            x0 = (INFO_W + ci * IMG_W + 0.11) / FW
            y0 = (row_bot + 0.44) / FH
            w_ = IMG_W_IN / FW
            h_ = IMG_H / FH

            ax_i = fig.add_axes([x0, y0, w_, h_])
            ax_i.imshow(img, aspect="equal")
            ax_i.set_xticks([]); ax_i.set_yticks([])
            bw = 2.8 if best else 0.9
            bc = col  if best else "#C8C8C8"
            for sp in ax_i.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(bc)
                sp.set_linewidth(bw)

            # CLIP score below image
            img_cx = (INFO_W + (ci + 0.5) * IMG_W) / FW
            label  = f"{clip:.4f}" + (" *" if best else "")
            fig.text(img_cx, (row_bot + 0.20) / FH, label,
                     ha="center", va="center",
                     fontsize=8.5,
                     color=col if best else "#888888",
                     fontweight="bold" if best else "normal")

        # Row separator line
        if ri < NROWS - 1:
            y_sep = (row_bot - GAP_V / 2) / FH
            fig.add_artist(plt.Line2D(
                [0.01, 0.99], [y_sep, y_sep],
                color="#DDDDDD", lw=0.7,
                transform=fig.transFigure))

    # Footer
    fig.text(0.5, 0.01,
             "* = highest CLIP score per diary row.  "
             "Bold border indicates the condition with best image–text alignment.",
             ha="center", va="bottom", fontsize=7.5, color="#888")

    plt.savefig(OUTPUTS / "figure_qualitative_gallery.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    print("✓ figure_qualitative_gallery.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — CLIP Heatmap (Mean row in grid + D9 anomaly annotation)
# ═══════════════════════════════════════════════════════════════════════════════
def make_clip_heatmap(scores):
    diary_ids = sorted(scores.keys(), key=lambda x: int(x.split("_")[1]))
    data      = np.array([[scores[d].get(c, np.nan) for c in CONDITIONS]
                           for d in diary_ids])
    col_means = np.nanmean(data, axis=0)

    # Append Mean row
    data_ext  = np.vstack([data, col_means])
    ylabels   = [d.replace("diary_", "D") for d in diary_ids] + ["Mean"]
    NROW      = len(ylabels)          # 11

    fig, ax = plt.subplots(figsize=(5.5, 5.8))

    im = ax.imshow(data_ext, cmap="YlGn", aspect="auto", vmin=0.185, vmax=0.310)

    # Axes ticks
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS],
                        fontsize=10.5, fontweight="bold")
    for ti, cond in enumerate(CONDITIONS):
        ax.xaxis.get_ticklabels()[ti].set_color(C_COLOR[cond])

    ax.set_yticks(range(NROW))
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.yaxis.get_ticklabels()[-1].set_fontweight("bold")   # Mean label bold

    ax.set_title("CLIP Score Heatmap  (10 Diaries × 3 Conditions)",
                  fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Cell value annotations
    for i in range(NROW):
        for j in range(3):
            val  = data_ext[i, j]
            if np.isnan(val): continue
            dark = val > 0.270
            bold = (i == NROW - 1)
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8.0,
                    fontweight="bold" if bold else "normal",
                    color="white" if dark else "#111")

    # Dashed separator above Mean row
    ax.axhline(NROW - 1.5, color="#444", lw=1.4, linestyle="--", zorder=5)

    # D9 anomaly — red rectangle highlight
    d9_idx = None
    for i, d in enumerate(diary_ids):
        if d == "diary_9":
            d9_idx = i
            break
    if d9_idx is not None:
        ax.add_patch(plt.Rectangle(
            (-0.5, d9_idx - 0.5), 3, 1,
            edgecolor="#CC0000", facecolor="none", lw=2.0, zorder=10))
        ax.text(3.08, d9_idx, "†",
                ha="left", va="center", fontsize=11,
                color="#CC0000", fontweight="bold", clip_on=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
    cbar.set_label("CLIP Cosine Similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_ylim(NROW - 0.5, -0.5)

    # D9 footnote
    fig.text(0.13, 0.015,
             "† D9: only diary where Partial > Full (CLIP anomaly)",
             ha="left", va="bottom", fontsize=7.5, color="#CC0000")

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    for ext in ("svg", "png"):
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kw["dpi"] = 200
        plt.savefig(OUTPUTS / f"figure_clip_heatmap.{ext}", format=ext, **kw)
        print(f"✓ figure_clip_heatmap.{ext}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — NEW: ΔCLIP Ablation Effect (Full − Baseline, Partial − Baseline)
# ═══════════════════════════════════════════════════════════════════════════════
def make_delta_plot(scores):
    diary_ids     = sorted(scores.keys(), key=lambda x: int(x.split("_")[1]))
    delta_full    = []
    delta_partial = []

    for d in diary_ids:
        bl = scores[d].get("condition_baseline", np.nan)
        pt = scores[d].get("condition_partial",  np.nan)
        fl = scores[d].get("condition_full",     np.nan)
        delta_full.append(fl - bl)
        delta_partial.append(pt - bl)

    x       = np.arange(len(diary_ids))
    w       = 0.36
    xlbls   = [d.replace("diary_", "D") for d in diary_ids]
    mean_fl = float(np.mean(delta_full))
    mean_pt = float(np.mean(delta_partial))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    bars_fl = ax.bar(x - w / 2, delta_full,    w,
                     color=C_FL, alpha=0.82, label="Full",    zorder=3)
    bars_pt = ax.bar(x + w / 2, delta_partial, w,
                     color=C_PT, alpha=0.82, label="Partial", zorder=3)

    # Zero reference
    ax.axhline(0, color="#333", lw=1.2, zorder=4)

    # Mean delta dashed lines
    ax.axhline(mean_fl, color=C_FL, lw=1.5, linestyle="--", alpha=0.65,
               label=f"Full mean  {mean_fl:+.4f}", zorder=5)
    ax.axhline(mean_pt, color=C_PT, lw=1.5, linestyle="--", alpha=0.65,
               label=f"Partial mean  {mean_pt:+.4f}", zorder=5)

    # Bar value labels
    for bar, vals, col in [(bars_fl, delta_full, C_FL),
                            (bars_pt, delta_partial, C_PT)]:
        for b, v in zip(bar, vals):
            ypos = v + 0.002 if v >= 0 else v - 0.005
            va   = "bottom" if v >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, ypos,
                    f"{v:+.3f}", ha="center", va=va,
                    fontsize=6.8, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_xlabel("Diary", fontsize=10.5)
    ax.set_ylabel("ΔCLIP Score  (condition − Baseline)", fontsize=10.5)
    ax.set_title("Ablation Effect: ΔCLIP Score per Diary",
                  fontsize=12, fontweight="bold", pad=8)

    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9.5)

    ax.legend(fontsize=8.5, framealpha=0.92,
               edgecolor="#CCCCCC", loc="lower left")

    fig.tight_layout()

    for ext in ("svg", "png"):
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kw["dpi"] = 200
        plt.savefig(OUTPUTS / f"figure_clip_delta.{ext}", format=ext, **kw)
        print(f"✓ figure_clip_delta.{ext}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    scores = load_clip_scores()
    make_pipeline_diagram()
    make_qualitative_gallery(scores)
    make_clip_heatmap(scores)
    make_delta_plot(scores)
    print("\n完成。所有图已保存至 outputs/")
