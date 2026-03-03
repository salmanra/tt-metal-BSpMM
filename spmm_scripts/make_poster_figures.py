#!/usr/bin/env python3
"""
make_poster_figures.py — Research-poster visualizations for BSR SpMM on Tenstorrent.

fig5_bsr_exec.png    — (a) BSR format  ·  (b) SpMM block-selection  ·  (c) Zigzag core grid
fig6_tensix_pipe.png — Single Tensix core pipeline: Reader0/1 → CBs → FPU → DRAM

Usage:
    python spmm_scripts/make_poster_figures.py [--out-dir spmm_plots]
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


# ── Palette ───────────────────────────────────────────────────────────────────
NAVY    = "#0D1B3E"
BLUE    = "#1565C0"
RED     = "#B71C1C"
GREEN   = "#1B5E20"
PURPLE  = "#4A148C"
ORANGE  = "#BF360C"
TEAL    = "#004D40"
AMBER   = "#E65100"
PINK    = "#880E4F"
INDIGO  = "#283593"
LIME    = "#33691E"
EMPTY   = "#ECEFF1"
GRIDC   = "#CFD8DC"
PANELBG = "#F5F7FC"

BLOCK_COLORS = [BLUE, RED, GREEN, PURPLE, ORANGE, TEAL, AMBER, PINK, INDIGO]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "figure.facecolor": "white",
})


# ── BSR example data ──────────────────────────────────────────────────────────
BR = 6   # block-rows in A
BC = 8   # block-cols in A
R  = 3   # visual cell size (rows per block)
C  = 3   # visual cell size (cols per block)

# Nonzero block positions — chosen to create visually interesting sparsity
NZ = sorted([
    (0, 0), (0, 5),
    (1, 3),
    # row 2: completely empty — demonstrates row skipping
    (3, 1), (3, 6),
    (4, 2), (4, 7),
    (5, 0), (5, 4),
])
NZ_COLOR = {pos: BLOCK_COLORS[i] for i, pos in enumerate(NZ)}


def bsr_arrays(nz, n_br):
    indptr = [0]
    indices = []
    for br in range(n_br):
        row_cols = sorted(c for (r, c) in nz if r == br)
        indices.extend(row_cols)
        indptr.append(indptr[-1] + len(row_cols))
    return indptr, indices

INDPTR, INDICES = bsr_arrays(NZ, BR)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def rect(ax, x, y, w, h, fc, ec=None, lw=1.0, alpha=1.0, zorder=2):
    p = Rectangle((x, y), w, h,
                  facecolor=fc, edgecolor=ec or fc,
                  linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def rounded_rect(ax, x, y, w, h, fc, ec=None, lw=1.0, alpha=1.0, zorder=2, r=0.06):
    """Rounded rectangle using FancyBboxPatch."""
    pad = r
    p = FancyBboxPatch((x + pad, y + pad), w - 2*pad, h - 2*pad,
                       boxstyle=f"round,pad={pad}",
                       facecolor=fc, edgecolor=ec or fc,
                       linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def ann(ax, src, dst, color, lw=2.0, rad=0.0, shrink=3, style="-|>"):
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(
                    arrowstyle=style, color=color, lw=lw,
                    shrinkA=shrink, shrinkB=shrink,
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=7)


def panel_label(ax, letter, x=-0.02, y=1.07):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=19,
            fontweight="bold", color=NAVY,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def title_bar(ax, txt, fontsize=14):
    ax.text(0.5, 1.04, txt, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold", color=NAVY)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — three-panel overview
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure5(out_dir: Path, clean=False):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")

    gs = GridSpec(1, 2, figure=fig,
                  width_ratios=[1.85, 2.35],
                  left=0.015, right=0.985,
                  top=0.82, bottom=0.04,
                  wspace=0.07)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    for ax in [ax_a, ax_b]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor("white" if clean else PANELBG)

    if not clean:
        fig.text(0.5, 0.980,
                 "Block Sparse Matrix Multiply (SpMM) on Tenstorrent Wormhole",
                 ha="center", va="top",
                 fontsize=23, fontweight="bold", color=NAVY)
        fig.text(0.5, 0.930,
                 "BSR format packs nonzero entries into dense R×C blocks  ·  "
                 "Each block selects a K-row band of B",
                 ha="center", va="top", fontsize=12.5, color="#455A64")

    _panel_a_bsr(ax_a, clean=clean)
    _panel_b_spmm(ax_b, clean=clean)

    suffix = "_clean" if clean else ""
    out = out_dir / f"fig5_bsr_exec{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Panel A: BSR Format ────────────────────────────────────────────────────────

def _panel_a_bsr(ax, clean=False):
    if not clean:
        title_bar(ax, "BSR Sparse Matrix Format")
        panel_label(ax, "a")

    W = BC * C
    H = BR * R
    ax.set_xlim(-1.8, W + 2.6)
    ax.set_ylim(-12.5, H + 2.5)
    ax.set_aspect("equal")

    # ── Block grid ────────────────────────────────────────────────────────────
    for br in range(BR):
        for bc in range(BC):
            x0 = bc * C
            y0 = (BR - 1 - br) * R
            pos = (br, bc)
            if pos in NZ_COLOR:
                col = NZ_COLOR[pos]
                rect(ax, x0, y0, C, R, fc=col, ec="white", lw=1.8, alpha=0.88, zorder=2)
                # Internal grid lines show tile structure
                for d in range(1, C):
                    ax.plot([x0+d, x0+d], [y0, y0+R],
                            color="white", lw=0.6, alpha=0.6, zorder=3)
                for d in range(1, R):
                    ax.plot([x0, x0+C], [y0+d, y0+d],
                            color="white", lw=0.6, alpha=0.6, zorder=3)
                idx = NZ.index(pos)
                ax.text(x0+C/2, y0+R/2, str(idx),
                        ha="center", va="center", fontsize=10.5,
                        color="white", fontweight="bold", zorder=4)
            else:
                rect(ax, x0, y0, C, R, fc=EMPTY, ec=GRIDC, lw=0.5, zorder=1)

    # Matrix border + label
    rect(ax, 0, 0, W, H, fc="none", ec=NAVY, lw=2.5, zorder=5)
    if not clean:
        ax.text(W/2, H + 1.6,
                f"A  [{BR*R} × {BC*C}]  —  {len(NZ)} of {BR*BC} blocks are nonzero",
                ha="center", va="center", fontsize=10, color=NAVY, style="italic")

    # Row annotations (density)
    if not clean:
        for br in range(BR):
            y = (BR - 1 - br) * R + R/2
            n = INDPTR[br+1] - INDPTR[br]
            tag = "empty" if n == 0 else f"{n}×"
            ax.text(-0.2, y, f"row {br}  ({tag})",
                    ha="right", va="center", fontsize=8, color="#546E7A")

    # ── Zoom inset: show R×C tile values in block 0 ───────────────────────────
    ZX, ZY = W + 0.4, (BR-1)*R + R*0.25     # inset top-right
    ZS = 1.3     # inset size
    col0 = NZ_COLOR[NZ[0]]
    # Dotted highlight around block 0 in main grid
    rect(ax, -0.06, (BR-1)*R - 0.06, C+0.12, R+0.12,
         fc="none", ec="goldenrod", lw=2.5, zorder=6)
    # Draw inset block with finer internal grid
    sub = 4    # show 4×4 sub-cells
    for si in range(sub):
        for sj in range(sub):
            sx = ZX + sj * ZS/sub
            sy = ZY + si * ZS/sub
            val = 0.4 + 0.15 * np.sin(si + sj)    # fake value pattern
            fc = mcolors.to_rgba(col0, alpha=max(0.25, val))
            rect(ax, sx, sy, ZS/sub - 0.04, ZS/sub - 0.04,
                 fc=col0, ec="white", lw=0.8, alpha=max(0.3, val), zorder=7)

    rect(ax, ZX - 0.05, ZY - 0.05, ZS + 0.10, ZS + 0.10,
         fc="none", ec="goldenrod", lw=1.8, zorder=8)
    if not clean:
        ax.text(ZX + ZS/2, ZY + ZS + 0.2,
                f"R×C = {R}×{C}\ndense block",
                ha="center", va="bottom", fontsize=7.5,
                color="goldenrod", fontweight="bold")
    # Zoom lines
    bx_tr = C
    by_tr = (BR-1)*R + R
    ax.annotate("", xy=(ZX, ZY + ZS),
                xytext=(bx_tr + 0.06, by_tr),
                arrowprops=dict(arrowstyle="-", color="goldenrod",
                                lw=1.0, linestyle="dashed"))
    ax.annotate("", xy=(ZX, ZY),
                xytext=(bx_tr + 0.06, (BR-1)*R),
                arrowprops=dict(arrowstyle="-", color="goldenrod",
                                lw=1.0, linestyle="dashed"))

    # ── Data structure arrays ─────────────────────────────────────────────────
    BW_ip = W / len(INDPTR)
    BW_id = W / len(INDICES)

    # indptr
    Y0 = -2.9
    ax.text(-0.2, Y0 + 0.5, "indptr :", ha="right", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    for i, v in enumerate(INDPTR):
        fc = "#BBDEFB" if i % 2 == 0 else "#E3F2FD"
        rect(ax, i*BW_ip, Y0, BW_ip - 0.12, 1.05, fc=fc, ec=BLUE, lw=1.0, zorder=3)
        ax.text(i*BW_ip + BW_ip/2, Y0 + 0.52, str(v),
                ha="center", va="center", fontsize=8.5, color=NAVY, fontweight="bold")
    if not clean:
        ax.text(W + 0.2, Y0 + 0.52,
                f"← {len(INDPTR)} entries\n(one per block-row + 1)",
                ha="left", va="center", fontsize=7.5, color="#546E7A")

    # indices
    Y1 = Y0 - 3.4
    ax.text(-0.2, Y1 + 0.5, "indices :", ha="right", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    for i, (v, (br, bc)) in enumerate(zip(INDICES, NZ)):
        col = NZ_COLOR[(br, bc)]
        rect(ax, i*BW_id, Y1, BW_id - 0.1, 1.05,
             fc=col, ec=col, alpha=0.88, lw=1.0, zorder=3)
        ax.text(i*BW_id + BW_id/2, Y1 + 0.52, str(v),
                ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")
    if not clean:
        ax.text(W + 0.2, Y1 + 0.52,
                f"← {len(INDICES)} entries\n(block-column index\nper nonzero block)",
                ha="left", va="center", fontsize=7.5, color="#546E7A")

    # data (symbolic)
    Y2 = Y1 - 3.4
    ax.text(-0.2, Y2 + 0.5, "data :", ha="right", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    for i, (br, bc) in enumerate(NZ):
        col = NZ_COLOR[(br, bc)]
        rect(ax, i*BW_id, Y2, BW_id - 0.1, 1.05,
             fc=col, ec=col, alpha=0.65, lw=0.8, zorder=3)
        ax.text(i*BW_id + BW_id/2, Y2 + 0.52, f"B{i}",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")
    if not clean:
        ax.text(W/2, Y2 - 1.0,
                f"Each Bᵢ: {R}×{C} = {R*C} fp16 values  "
                f"({len(NZ)} blocks total, {len(NZ)*R*C} nonzero scalars)",
                ha="center", va="top", fontsize=9, color="#546E7A", style="italic")


# ── Panel B: SpMM computation ─────────────────────────────────────────────────

def _panel_b_spmm(ax, clean=False):
    if not clean:
        title_bar(ax, "SpMM: each nnz block selects a B-matrix strip")
        panel_label(ax, "b")

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, 11.3)

    # Matrix boundaries — B and C have equal width (3.05 each)
    Ax0, Ax1 = 0.25, 1.85
    Bx0, Bx1 = 2.90, 5.95
    Cx0, Cx1 = 6.75, 9.80
    My0, My1 = 0.9, 10.8

    Aw = Ax1 - Ax0;  Bw = Bx1 - Bx0;  Cw = Cx1 - Cx0
    Ah = My1 - My0;  Bh = My1 - My0;  Ch = My1 - My0
    bh_A = Ah / BR
    bw_A = Aw / BC
    bh_B = Bh / BC   # K-dimension: BC block-row bands in B (match A block-col count)
    bw_B = Bw / BC   # N-dimension: BC output-col strips in B (visual only)
    bh_C = Ch / BR

    # ── Draw A ────────────────────────────────────────────────────────────────
    rect(ax, Ax0, My0, Aw, Ah, fc="#F3F4F8", ec=GRIDC, lw=1.0, zorder=1)
    for br in range(BR):
        for bc in range(BC):
            pos = (br, bc)
            if pos in NZ_COLOR:
                col = NZ_COLOR[pos]
                x0 = Ax0 + bc * bw_A
                y0 = My1 - (br + 1) * bh_A
                rect(ax, x0, y0, bw_A, bh_A, fc=col, ec="white", lw=1.2,
                     alpha=0.88, zorder=3)
    rect(ax, Ax0, My0, Aw, Ah, fc="none", ec=NAVY, lw=2.2, zorder=4)
    ax.text((Ax0+Ax1)/2, My1 + 0.35, "A  (sparse BSR)",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text((Ax0+Ax1)/2, My0 - 0.35, "M × K",
            ha="center", fontsize=9, color="#78909C", style="italic")

    # ── Draw B — K×N grid: vertical output-col strips + horizontal K-row bands ──
    # Background: alternating vertical strips (N-dimension, output columns)
    for j in range(BC):
        x0 = Bx0 + j * bw_B
        fc = "#F1F8E9" if j % 2 == 0 else "#E8F5E9"
        rect(ax, x0, My0, bw_B, Bh, fc=fc, ec="none", lw=0, zorder=1)
    # Faint horizontal lines at K-block-row boundaries (K-dimension structure)
    for i in range(1, BC):
        y_line = My1 - i * bh_B
        ax.plot([Bx0, Bx1], [y_line, y_line], lw=0.6, color="#B0BEC5", zorder=2)
    # Faint vertical lines (N-dimension output-col structure)
    for j in range(1, BC):
        x_line = Bx0 + j * bw_B
        ax.plot([x_line, x_line], [My0, My1], lw=0.4, color="#C8E6C9", zorder=2)
    rect(ax, Bx0, My0, Bw, Bh, fc="none", ec=NAVY, lw=2.2, zorder=4)
    ax.text((Bx0+Bx1)/2, My1 + 0.35, "B  (dense)",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text((Bx0+Bx1)/2, My0 - 0.35, "K × N",
            ha="center", fontsize=9, color="#78909C", style="italic")

    # ── Draw C ────────────────────────────────────────────────────────────────
    rect(ax, Cx0, My0, Cw, Ch, fc="#FFF9C4", ec=NAVY, lw=2.2, zorder=1)
    ax.text((Cx0+Cx1)/2, My1 + 0.35, "C  (dense output)",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text((Cx0+Cx1)/2, My0 - 0.35, "M × N",
            ha="center", fontsize=9, color="#78909C", style="italic")

    # Operators
    ax.text((Ax1+Bx0)/2, (My0+My1)/2, "×",
            ha="center", va="center", fontsize=26, color=NAVY, fontweight="bold")
    ax.text((Bx1+Cx0)/2, (My0+My1)/2, "→",
            ha="center", va="center", fontsize=26, color=NAVY, fontweight="bold")

    # ── Highlight block-row 3 (blocks at cols 1 and 6) ───────────────────────
    FOCUS = 3
    focus_cols = [bc for (br, bc) in NZ if br == FOCUS]

    # Highlighted row band in A
    y_a = My1 - (FOCUS + 1) * bh_A
    a_mid_y = y_a + bh_A / 2
    rect(ax, Ax0 - 0.05, y_a - 0.05, Aw + 0.10, bh_A + 0.10,
         fc="none", ec="goldenrod", lw=3.0, zorder=6)
    if not clean:
        ax.text(Ax0 - 0.10, a_mid_y, f"row {FOCUS}",
                ha="right", va="center", fontsize=9.5,
                color="goldenrod", fontweight="bold")

    # Highlighted output row in C
    y_c = My1 - (FOCUS + 1) * bh_C
    rect(ax, Cx0, y_c, Cw, bh_C, fc="gold", ec="goldenrod",
         lw=2.5, alpha=0.75, zorder=3)
    ax.text((Cx0+Cx1)/2, y_c + bh_C/2, "Σ",
            ha="center", va="center", fontsize=16,
            color=NAVY, fontweight="bold", zorder=4)

    # For each focus block: highlight the K-row band in B that aligns with
    # the A block's column index (spans full N width — fetched for every output col)
    for i, bc in enumerate(focus_cols):
        col = NZ_COLOR[(FOCUS, bc)]

        # B K-row band: full-width horizontal stripe at K-block-row bc
        y_row = My1 - (bc + 1) * bh_B
        y_row_mid = y_row + bh_B / 2
        rect(ax, Bx0, y_row, Bw, bh_B, fc=col, ec=col, alpha=0.72, lw=1.5, zorder=3)
        # Label on the left side of B
        if not clean:
            ax.text(Bx0 - 0.12, y_row_mid,
                    f"row-blk {bc}", ha="right", va="center",
                    fontsize=8.5, color=col, fontweight="bold")

        # Arrow A → B: from right edge of A row → left edge of B K-row stripe
        # bc=1 stripe is high in B, bc=6 is low → arrows diverge up/down from A row
        rad_ab = 0.22 if i == 0 else -0.22
        ann(ax, (Ax1 + 0.05, a_mid_y),
                (Bx0 - 0.05, y_row_mid),
                color=col, lw=2.2, rad=rad_ab, shrink=2)

        # Arrow B → C: from right edge of B K-row stripe → left of C output row
        # Both converge on the same output row in C
        rad_bc = -0.22 if i == 0 else 0.22
        ann(ax, (Bx1 + 0.05, y_row_mid),
                (Cx0 - 0.05, y_c + bh_C/2),
                color=col, lw=2.2, rad=rad_bc, shrink=2)

    if not clean:
        # Key insight annotation
        ax.text(5.0, -0.45,
                "Each nonzero A block at column bc fetches K-row block bc of B"
                "  (one thin horizontal band per nnz block, spanning all N output columns).\n"
                "Bands are multiplied and accumulated (Σ) into the output row of C.\n"
                "Skipping zero block-rows avoids wasted computation and DRAM traffic.",
                ha="center", va="top", fontsize=9.5, color="#37474F", style="italic")

        # ── Mini legend (only the 2 blocks active in the focused row) ────────────
        focus_nz = [(i, pos) for i, pos in enumerate(NZ) if pos[0] == FOCUS]
        leg = [mpatches.Patch(fc=NZ_COLOR[pos],
                              label=f"Block {i}: A[row {pos[0]}, col {pos[1]}]  →  B K-row block {pos[1]}")
               for i, pos in focus_nz]
        leg.append(mpatches.Patch(fc="gold", ec="goldenrod", lw=1.5,
                                   label=f"C output row {FOCUS}  (Σ of both blocks)"))
        ax.legend(handles=leg, loc="upper right",
                  bbox_to_anchor=(1.0, 0.30), fontsize=8.0,
                  framealpha=0.95, ncol=1, edgecolor=GRIDC,
                  title=f"Focused on A row {FOCUS}  ({len(focus_nz)} nnz blocks)",
                  title_fontsize=8.5)


# ── Panel C: zigzag load-balancing before/after ───────────────────────────────

def _panel_c_loadbalance(ax, show_label=True, clean=False):
    if not clean:
        title_bar(ax, "Tenstorrent Core Grid — Zigzag Load Balancing")
        if show_label:
            panel_label(ax, "c")

    # ── Simulation parameters ─────────────────────────────────────────────────
    # 16 nonzero block-rows with varying density, distributed over 8 core-rows
    # (each core-row handles 2 nnz rows = num_iters_y=2)
    NNZ    = 16
    NCORES_R = 8
    NCORES_C = 5       # display cols (per half)
    NITERS   = NNZ // NCORES_R   # = 2

    # Densities (nnz blocks per row) — decreasing, simulating realistic sparsity
    rng = np.random.default_rng(7)
    raw = sorted(rng.integers(2, 22, size=NNZ), reverse=True)
    densities = list(raw)   # [21, 19, 17, 16, 14, 13, 11, 10, 9, 8, 7, 6, 4, 3, 2, 1]

    # Naive: core_row i gets rows [2i, 2i+1] in sorted order (heavy top)
    naive_work = np.array([densities[NITERS*i] + densities[NITERS*i+1]
                           for i in range(NCORES_R)], dtype=float)

    # Zigzag: forward (perm[0..7]) + backward (perm[15..8])
    zigzag_work = np.array([densities[i] + densities[NNZ-1-i]
                            for i in range(NCORES_R)], dtype=float)

    # ── Colormap: workload intensity (light = low, dark = high) ──────────────
    cmap = plt.cm.YlOrRd
    vmax = naive_work.max()

    CS  = 0.76   # cell size
    GAP = 0.08   # gap between cells
    GW  = NCORES_C * (CS + GAP)
    GH  = NCORES_R * (CS + GAP)

    # Horizontal layout: [naive] [gap] [zigzag]
    # X_naive shifted right to leave room for input-rows column + DRAM arrow
    X_naive  = 1.80
    X_zigzag = X_naive + GW + 3.0   # wider gap for DRAM arrows & explanation box
    Y0       = 0.50

    ax.set_xlim(-0.2, X_zigzag + GW + 3.5)
    ax.set_ylim(-4.5, Y0 + GH + 3.5)

    # Pre-compute imbalance strings (can't use loop variable in list literal)
    naive_imb  = f"Imbalance: {naive_work.max()/naive_work.min():.1f}×"
    zigzag_imb = f"Imbalance: {zigzag_work.max()/zigzag_work.min():.2f}×  ✓"

    # ── Draw both grids ───────────────────────────────────────────────────────
    for grid_x, work_per_row, subtitle, imb_note in [
        (X_naive,  naive_work,  "Naïve  (input order)",   naive_imb),
        (X_zigzag, zigzag_work, "Sweep  (load-balanced)", zigzag_imb),
    ]:
        # Subtitle + imbalance note
        ax.text(grid_x + GW/2, Y0 + GH + 0.50, subtitle,
                ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=NAVY)

        for row in range(NCORES_R):
            work = work_per_row[row]
            fc   = cmap(work / vmax)

            for col in range(NCORES_C):
                cx = grid_x + col * (CS + GAP)
                cy = Y0 + (NCORES_R - 1 - row) * (CS + GAP)
                rounded_rect(ax, cx, cy, CS, CS, fc=fc, ec="white",
                             lw=1.5, zorder=3, r=0.04)

                # Show work count in middle column only
                if col == NCORES_C // 2:
                    txt_c = "white" if work/vmax > 0.55 else NAVY
                    ax.text(cx + CS/2, cy + CS/2,
                            f"{int(work)}",
                            ha="center", va="center", fontsize=7.5,
                            color=txt_c, fontweight="bold", zorder=4)

            # Row label on right side
            if not clean:
                ax.text(grid_x + GW + 0.12,
                        Y0 + (NCORES_R - 1 - row) * (CS + GAP) + CS/2,
                        f"core-row {row}",
                        ha="left", va="center", fontsize=7.2, color="#78909C")

        # Grid border
        gw = NCORES_C * (CS + GAP) - GAP
        gh = NCORES_R * (CS + GAP) - GAP
        rect(ax, grid_x, Y0, gw, gh, fc="none", ec=NAVY, lw=2.3, zorder=5)

        # ── Input rows column (left of core grid) ─────────────────────────────
        IR_w   = 0.65
        IR_gap = 0.08   # gap: right edge of input rows → left edge of grid
        IR_h   = GH / NNZ
        IR_x0  = grid_x - IR_gap - IR_w   # left edge of input rows
        IR_x1  = grid_x - IR_gap           # right edge of input rows

        # Assignment: which input rows go to which core row?
        if "Naïve" in subtitle:
            core_to_rows = {i: [NITERS * i, NITERS * i + 1] for i in range(NCORES_R)}
        else:  # Zigzag: pair row i with its mirror NNZ-1-i
            core_to_rows = {i: [i, NNZ - 1 - i] for i in range(NCORES_R)}

        # Draw input row cells (colored by density, same heatmap)
        for inp_row in range(NNZ):
            iy = Y0 + GH - (inp_row + 1) * IR_h
            rect(ax, IR_x0, iy, IR_w, IR_h - 0.01,
                 fc=cmap(densities[inp_row] / vmax), ec="white", lw=0.5, alpha=0.9, zorder=3)
        # Thin border around the column
        rect(ax, IR_x0, Y0, IR_w, GH - GAP, fc="none", ec="#B0BEC5", lw=0.8, zorder=4)

        # Connection lines: each input row → its assigned core row
        for core_row_idx, inp_rows_list in core_to_rows.items():
            core_y = Y0 + (NCORES_R - 1 - core_row_idx) * (CS + GAP) + CS / 2
            for inp_row in inp_rows_list:
                inp_y = Y0 + GH - (inp_row + 0.5) * IR_h
                ax.plot([IR_x1, grid_x], [inp_y, core_y],
                        color=cmap(densities[inp_row] / vmax),
                        lw=0.9, alpha=0.55, zorder=2)

        ax.text(IR_x0 + IR_w / 2, Y0 + GH + 0.20, "Input rows",
                ha="center", va="bottom", fontsize=7.5, color="#546E7A")

        if not clean:
            # DRAM arrow — points to left edge of input rows column
            ax.annotate("", xy=(IR_x0, Y0 + GH*0.5),
                        xytext=(IR_x0 - 0.55, Y0 + GH*0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INDIGO, lw=2.0))
            ax.text(IR_x0 - 0.58, Y0 + GH*0.5, "DRAM",
                    ha="right", va="center", fontsize=9,
                    fontweight="bold", color=INDIGO)

            # Imbalance badge below
            badge_col = RED if "Naïve" in subtitle else GREEN
            ax.text(grid_x + GW/2, Y0 - 0.45, imb_note,
                    ha="center", va="top", fontsize=10.5,
                    fontweight="bold", color=badge_col,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor=badge_col, lw=1.5))

    # ── Work bar charts below grids ───────────────────────────────────────────
    bar_y0   = -4.0
    bar_h    = 2.8
    bar_bw   = GW / NCORES_R * 0.72
    bar_gap  = GW / NCORES_R

    for grid_x, work_per_row in [(X_naive, naive_work), (X_zigzag, zigzag_work)]:
        ax.axhline(bar_y0, xmin=0, xmax=1, color=GRIDC, lw=0.5)
        for row in range(NCORES_R):
            bx = grid_x + row * bar_gap + bar_gap*0.14
            bh = bar_h * work_per_row[row] / vmax
            fc = cmap(work_per_row[row] / vmax)
            rect(ax, bx, bar_y0, bar_bw, bh, fc=fc, ec="white", lw=0.8, alpha=0.9)

        # Axis for bar chart
        rect(ax, grid_x, bar_y0, GW, 0.02, fc=NAVY, ec=NAVY, lw=0)
        ax.text(grid_x + GW/2, bar_y0 - 0.35, "Core-row  →",
                ha="center", va="top", fontsize=8, color="#546E7A")
        ax.text(grid_x - 0.15, bar_y0 + bar_h/2, "Total\nwork",
                ha="right", va="center", fontsize=7.5,
                color="#546E7A", rotation=90)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    if not clean:
        cbar_x  = X_zigzag + GW + 1.55
        cbar_y  = Y0
        cbar_w  = 0.28
        cbar_h  = GH
        N_STEPS = 20
        for i in range(N_STEPS):
            cy = cbar_y + i / N_STEPS * cbar_h
            ch = cbar_h / N_STEPS
            fc = cmap((i + 0.5) / N_STEPS)
            rect(ax, cbar_x, cy, cbar_w, ch, fc=fc, ec=fc, zorder=2)
        rect(ax, cbar_x, cbar_y, cbar_w, cbar_h, fc="none", ec=NAVY, lw=1.5, zorder=3)
        ax.text(cbar_x + cbar_w/2, cbar_y - 0.2, "light\n(idle)",
                ha="center", va="top", fontsize=7.5, color="#546E7A")
        ax.text(cbar_x + cbar_w/2, cbar_y + cbar_h + 0.1, "dark\n(heavy)",
                ha="center", va="bottom", fontsize=7.5, color="#546E7A")
        ax.text(cbar_x + cbar_w + 0.15, cbar_y + cbar_h/2,
                "Workload\nper core-row\n(blocks to compute)",
                ha="left", va="center", fontsize=8, color="#37474F", rotation=0)

    # ── Explanation text ──────────────────────────────────────────────────────
    if not clean:
        sep_x = (X_naive + GW + X_zigzag) / 2
        ax.text(sep_x, Y0 + GH/2,
                "Zigzag:\nheavy + light\nrow paired\nper core",
                ha="center", va="center", fontsize=9, color=TEAL,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E0F2F1",
                          edgecolor=TEAL, lw=1.5, alpha=0.95))
        ann(ax, (sep_x - 0.55, Y0 + GH*0.6), (X_naive + GW + 0.05, Y0 + GH*0.6),
            color=RED, lw=1.5, rad=0)
        ann(ax, (sep_x + 0.55, Y0 + GH*0.6), (X_zigzag - 0.05, Y0 + GH*0.6),
            color=GREEN, lw=1.5, rad=0)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Tensix core pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure5c(out_dir: Path, clean=False):
    """Panel C (zigzag load balancing) as a standalone poster figure."""
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("white" if clean else PANELBG)

    _panel_c_loadbalance(ax, show_label=False, clean=clean)

    suffix = "_clean" if clean else ""
    out = out_dir / f"fig5c_loadbalance{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_figure6(out_dir: Path, clean=False):
    fig, ax = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white" if clean else PANELBG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_xlim(0, 22)
    ax.set_ylim(-0.8, 10.5)

    if not clean:
        fig.suptitle(
            "Tenstorrent Tensix Core — BSR SpMM Execution Pipeline",
            fontsize=20, fontweight="bold", color=NAVY, y=0.99)
        ax.text(11, 10.1,
                "Three RISC-V processors coordinate via double-buffered circular buffers (CBs).  "
                "I/O and compute overlap so cores stay busy.",
                ha="center", va="top", fontsize=12, color="#455A64")

    _draw_tensix_pipeline(ax, clean=clean)

    suffix = "_clean" if clean else ""
    out = out_dir / f"fig6_tensix_pipe{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _draw_tensix_pipeline(ax, clean=False):
    BH  = 1.15   # box height
    BW  = 2.05   # box width

    # Y lanes: more vertical space between A, B, and output
    Y_A   = 7.5    # sparse A lane (Reader 0)
    Y_B   = 4.8    # dense B lane (Reader 1)
    Y_OUT = 1.8    # output lane

    # ── DRAM left ─────────────────────────────────────────────────────────────
    for dy, lbl, col in [(Y_A - BH/2, "DRAM\nSparse A\n(BSR blocks)", INDIGO),
                          (Y_B - BH/2, "DRAM\nDense B\n(full matrix)",  TEAL)]:
        rounded_rect(ax, 0.18, dy, BW, BH, fc=col, ec="white",
                     lw=1.5, alpha=0.88, zorder=3, r=0.05)
        ax.text(0.18 + BW/2, dy + BH/2, lbl,
                ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold", zorder=4)

    # ── Reader 0 ──────────────────────────────────────────────────────────────
    R0x = 3.15
    rounded_rect(ax, R0x, Y_A - BH/2, BW, BH, fc=BLUE, ec="white",
                 lw=1.5, alpha=0.90, zorder=3, r=0.05)
    ax.text(R0x + BW/2, Y_A,
            "Reader 0  (BRISC)\n— reads sparse A\n— writes output C",
            ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=4)

    # ── Reader 1 ──────────────────────────────────────────────────────────────
    rounded_rect(ax, R0x, Y_B - BH/2, BW, BH, fc=GREEN, ec="white",
                 lw=1.5, alpha=0.90, zorder=3, r=0.05)
    ax.text(R0x + BW/2, Y_B,
            "Reader 1  (NCRISC)\n— reads dense B\n— feeds CB c₁",
            ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=4)

    # ── Circular buffers ──────────────────────────────────────────────────────
    CB_x = 6.15
    CB_w = 1.42
    CB_h = 1.05

    cb_info = [
        (Y_A - CB_h/2,         "CB c₀\nA tiles  ×2",  BLUE,   True),
        (Y_B - CB_h/2,         "CB c₁\nB tiles  ×2",  GREEN,  True),
        (Y_OUT + 0.60,         "CB c₁₆\noutput  ×1",  RED,    False),
        (Y_OUT + 0.60 + CB_h + 0.30, "CB c₂₄\nspill ×1",  AMBER,  False),
    ]
    for cy, lbl, col, dbl in cb_info:
        if dbl:
            rounded_rect(ax, CB_x+0.09, cy+0.09, CB_w, CB_h,
                         fc=col, ec="white", lw=0.5, alpha=0.32, zorder=3, r=0.04)
        rounded_rect(ax, CB_x, cy, CB_w, CB_h,
                     fc=col, ec="white", lw=1.5, alpha=0.87, zorder=4, r=0.04)
        ax.text(CB_x + CB_w/2, cy + CB_h/2, lbl,
                ha="center", va="center", fontsize=8.2,
                color="white", fontweight="bold", zorder=5)

    # L1 SRAM box (encloses all CBs and FPU)
    L1x, L1y, L1w, L1h = 5.75, 0.90, 5.85, 8.50
    rect(ax, L1x, L1y, L1w, L1h, fc="none", ec="#90A4AE", lw=2.8, zorder=1)
    # Label: sits ON the top-right portion of the border line, white bbox masks the line behind it
    # (classic circuit-diagram "box title" style — avoids all interior callout overlap)
    ax.text(L1x + L1w - 0.20, L1y + L1h,
            "  L1 SRAM  (1.5 MB / core)  ",
            ha="right", va="center", fontsize=9.5, fontweight="bold", color="#546E7A",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=1.0), zorder=6)

    # ── Compute FPU ───────────────────────────────────────────────────────────
    FX, FY, FW, FH = 8.55, 3.50, 2.50, 3.60
    rounded_rect(ax, FX, FY, FW, FH, fc=RED, ec="white", lw=2.2,
                 alpha=0.90, zorder=3, r=0.09)
    ax.text(FX + FW/2, FY + FH*0.74,
            "Compute\n(TRISC / FPU)",
            ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold", zorder=4)
    ax.text(FX + FW/2, FY + FH*0.32,
            "matmul_tiles()\n32×32 fp16 tiles\naccumulate in DST",
            ha="center", va="center", fontsize=8.0,
            color="#FFCDD2", zorder=4, style="italic")

    # ── DRAM output ───────────────────────────────────────────────────────────
    OUTx = 12.10
    rounded_rect(ax, OUTx, Y_OUT - BH/2, BW, BH, fc=PINK, ec="white",
                 lw=1.5, alpha=0.90, zorder=3, r=0.05)
    ax.text(OUTx + BW/2, Y_OUT,
            "DRAM\nOutput C",
            ha="center", va="center", fontsize=9.5,
            color="white", fontweight="bold", zorder=4)

    # ── Arrows ────────────────────────────────────────────────────────────────
    ann(ax, (0.18+BW, Y_A), (R0x, Y_A), BLUE, lw=2.0)
    ann(ax, (0.18+BW, Y_B), (R0x, Y_B), GREEN, lw=2.0)
    ann(ax, (R0x+BW, Y_A), (CB_x, cb_info[0][0]+CB_h/2), BLUE, lw=2.0)
    ann(ax, (R0x+BW, Y_B), (CB_x, cb_info[1][0]+CB_h/2), GREEN, lw=2.0)
    ann(ax, (CB_x+CB_w, cb_info[0][0]+CB_h/2), (FX, FY+FH*0.88), BLUE, lw=2.0)
    ann(ax, (CB_x+CB_w, cb_info[1][0]+CB_h/2), (FX, FY+FH*0.52), GREEN, lw=2.0)
    # Compute → CB c_24 (spill)
    c24_cy = cb_info[3][0]
    ann(ax, (FX + FW*0.25, FY+0.08), (CB_x+CB_w, c24_cy+CB_h/2),
        AMBER, lw=1.8, rad=-0.30)
    # CB c_24 → Compute (reload)
    ann(ax, (CB_x+CB_w, c24_cy+CB_h*0.7), (FX + FW*0.05, FY+0.30),
        "#FF8F00", lw=1.8, rad=0.35)
    # Compute → CB c_16
    c16_cy = cb_info[2][0]
    ann(ax, (FX, FY+FH*0.08), (CB_x+CB_w, c16_cy+CB_h/2),
        RED, lw=2.0, rad=0.25)
    # CB c_16 → DRAM output
    ann(ax, (CB_x+CB_w, c16_cy+CB_h/2), (OUTx, Y_OUT), PINK, lw=2.0, rad=-0.20)

    # ── Double-buffering Gantt timeline ───────────────────────────────────────
    TX0  = 14.80    # timeline x start
    TXW  = 6.80     # total width
    TH   = 0.80     # bar height (generous)
    TGAP = 0.28     # gap between lanes
    TY_FPU  = 0.50
    TY_R1   = TY_FPU + TH + TGAP
    TY_R0   = TY_R1  + TH + TGAP

    if not clean:
        ax.text(TX0 + TXW/2, TY_R0 + TH + 0.55,
                "Double-buffering timeline (3 tiles shown)",
                ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=NAVY)

    # Each (label, color, lane_y, slot_start)
    N_SLOTS = 6
    sw = TXW / (N_SLOTS + 0.5)   # slot width

    timeline_bars = [
        ("Read A₀",   BLUE,  TY_R0, 0),
        ("Read A₁",   BLUE,  TY_R0, 1),
        ("Read A₂",   BLUE,  TY_R0, 2),
        ("Read B₀",   GREEN, TY_R1, 0),
        ("Read B₁",   GREEN, TY_R1, 1),
        ("Read B₂",   GREEN, TY_R1, 2),
        ("Compute₀",  RED,   TY_FPU, 1),
        ("Compute₁",  RED,   TY_FPU, 2),
        ("Compute₂",  RED,   TY_FPU, 3),
    ]
    for lbl, col, ty, slot in timeline_bars:
        x = TX0 + slot * sw
        rounded_rect(ax, x + 0.07, ty + 0.06, sw - 0.14, TH - 0.12,
                     fc=col, ec="white", lw=1.0, alpha=0.87, zorder=3, r=0.05)
        ax.text(x + sw/2, ty + TH/2, lbl,
                ha="center", va="center", fontsize=8.0,
                color="white", fontweight="bold", zorder=4)

    # Lane labels
    for ty, lbl, col in [(TY_R0, "Reader 0", BLUE),
                          (TY_R1, "Reader 1", GREEN),
                          (TY_FPU, "FPU",      RED)]:
        ax.text(TX0 - 0.18, ty + TH/2, lbl,
                ha="right", va="center", fontsize=9.0,
                color=col, fontweight="bold")

    # Overlap highlight: show that Compute₀ overlaps Read A₁ / Read B₁
    ax.axvspan(TX0 + 1*sw, TX0 + 2*sw, ymin=0, ymax=1,
               alpha=0.06, color="gold", zorder=0)
    if not clean:
        ax.text(TX0 + 1.5*sw, TY_R0 + TH + 0.12,
                "parallel", ha="center", va="bottom",
                fontsize=7.5, color="goldenrod", fontweight="bold")

        # Time arrow
        ax.annotate("", xy=(TX0 + TXW + 0.1, TY_FPU - 0.22),
                    xytext=(TX0, TY_FPU - 0.22),
                    arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5))
        ax.text(TX0 + TXW + 0.2, TY_FPU - 0.22, "Time →",
                ha="left", va="center", fontsize=9, color="#546E7A")
        ax.text(TX0 + TXW/2, TY_FPU - 0.60,
                "Reader prefetches next tile while FPU computes current  —  hides DRAM latency",
                ha="center", va="top", fontsize=8.5, color="#37474F", style="italic")

        # ── Step callouts (well-separated from boxes) ─────────────────────────────
        step_cfg = [
            (1.15,  5.85, "① Fetch\nBSR block\nfrom DRAM",             BLUE),
            (1.15,  3.55, "② Fetch\nB-strip\nfrom DRAM",               GREEN),
            (6.85,  9.30, "③ Double-buffered CBs\noverlap I/O + compute", "#37474F"),
            (9.80,  8.80, "④ FPU: outer-product\ntile matmul per\n32×32 block", RED),
            (9.80,  2.10, "⑤ Partial sums spill\nto c₂₄; reloaded\nfor next nnz block", AMBER),
            (13.15, 3.80, "⑥ Final block:\nReader 0 drains c₁₆\nand writes C to DRAM", PINK),
        ]
        for sx, sy, stxt, col in step_cfg:
            ax.text(sx, sy, stxt, ha="center", va="center",
                    fontsize=8.2, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                              edgecolor=col, lw=1.6, alpha=0.96),
                    zorder=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate research-poster figures for BSR SpMM on Tenstorrent")
    parser.add_argument("--out-dir", type=Path, default=Path("spmm_plots"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    make_figure5(args.out_dir)
    make_figure5c(args.out_dir)
    make_figure6(args.out_dir)
    make_figure5(args.out_dir, clean=True)
    make_figure5c(args.out_dir, clean=True)
    make_figure6(args.out_dir, clean=True)


if __name__ == "__main__":
    main()
