"""Figures for the report. Matplotlib only, no seaborn, no style dependencies."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#11151c"
ACCENT = "#bd3039"   # Red Sox red
MUTED = "#8a9099"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(True, alpha=0.18, linewidth=0.7)


def calibration(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=170)
    lo = float(min(table["predicted"].min(), table["observed"].min()))
    hi = float(max(table["predicted"].max(), table["observed"].max()))
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=MUTED, lw=1, label="perfect calibration")
    ax.plot(table["predicted"], table["observed"], "o-", color=ACCENT, lw=1.6, ms=6, label="model")
    _style(ax, "Calibration — predicted vs observed whiff rate", "Predicted whiff rate", "Observed whiff rate")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def whiff_by_velo(df: pd.DataFrame, out: Path) -> None:
    """Observed and modeled whiff rate across velocity, four-seamers only."""
    ff = df[df["pitch_type"] == "FF"].copy()
    ff = ff[(ff["release_speed"] >= 88) & (ff["release_speed"] <= 102)]
    ff["bucket"] = (ff["release_speed"] // 1) * 1
    g = ff.groupby("bucket").agg(
        observed=("is_whiff", "mean"), expected=("p_stuff", "mean"), n=("is_whiff", "size")
    ).reset_index()
    g = g[g["n"] >= 400]

    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=170)
    ax.plot(g["bucket"], g["observed"], "o-", color=ACCENT, lw=1.8, ms=5, label="observed")
    ax.plot(g["bucket"], g["expected"], "s--", color=INK, lw=1.4, ms=4, alpha=0.75, label="model (stuff only)")
    _style(ax, "Four-seam whiff rate by velocity", "Release speed (mph)", "Whiff rate per swing")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def stuff_vs_outcome(grades: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=170)
    ax.scatter(grades["stuff_plus"], grades["actual_whiff_rate"],
               s=16, color=ACCENT, alpha=0.55, edgecolor="none")
    if len(grades) > 2:
        b, a = np.polyfit(grades["stuff_plus"], grades["actual_whiff_rate"], 1)
        xs = np.linspace(grades["stuff_plus"].min(), grades["stuff_plus"].max(), 50)
        ax.plot(xs, a + b * xs, color=INK, lw=1.3, alpha=0.8)
    r = grades["stuff_plus"].corr(grades["actual_whiff_rate"])
    _style(ax, f"Stuff+ vs actual whiff rate  (r = {r:.2f})", "Stuff+ (100 = league average)", "Actual whiff rate")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def movement_map(df: pd.DataFrame, out: Path) -> None:
    """Expected whiff across the movement plane, by pitch family."""
    fams = [("FF", "Four-seam"), ("SL", "Slider"), ("CH", "Changeup"), ("CU", "Curveball")]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4), dpi=170)
    for (pt, label), ax in zip(fams, axes.ravel()):
        sub = df[df["pitch_type"] == pt]
        if len(sub) < 500:
            ax.set_visible(False); continue
        sc = ax.scatter(sub["pfx_x_mir"] * 12, sub["pfx_z"] * 12, c=sub["p_stuff"],
                        s=2, cmap="RdYlBu_r", alpha=0.45, edgecolor="none")
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        _style(ax, f"{label} — expected whiff by movement", "Horizontal break (in, RHP frame)", "Induced vertical break (in)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
