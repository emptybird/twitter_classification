"""
CSCI946 Assignment 2 - Data preparation figures
Author: Tianyu He (9495204)

Run after prepare_data.py:   python make_figures.py
Writes three PNGs to output/figures/.

Each figure exists to prove one sentence in the Data Preparation section of
the report. A figure that proves nothing is not drawn.

  fig1_skew_log_transform.png  "the count columns are severely right skewed,
                                so a log1p transform was applied"
  fig2_missing_values.png      "these columns were dropped because too much
                                of them is missing"
  fig3_rows_per_account.png    "the file is not one row per account, so a
                                random train/test split leaks"

One hue is used throughout; a second hue appears only where two groups must
be told apart, and then a legend is always present. Every axis starts at
zero, so no difference is visually exaggerated.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# --- palette (light surface, for a printed report) ------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})
SAVE = dict(dpi=200, bbox_inches="tight")


def title(ax, text, sub=None):
    """Title and subtitle offset in points, so spacing does not depend on
    how tall the axes happens to be."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 24 if sub else 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=11, color=INK,
                fontweight="medium", annotation_clip=False)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=INK2,
                    annotation_clip=False)


def compact(x, _pos=None):
    """0 / 50k / 1.2M - short enough that axis labels never collide."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if x >= 1_000:
        return f"{x / 1_000:g}k"
    return f"{x:g}"


def load_raw(apply_row_filter=True):
    """Load the source CSV.

    apply_row_filter mirrors step 2 of prepare_data.py (drop the 97 rows with
    profile_yn == 'no'). The distribution figures must be computed on the same
    19,953 rows the pipeline actually transformed, otherwise the skewness
    printed on a figure would not match the one in data_quality_report.txt.
    """
    for c in ["../A2_2026_Released", "A2_2026_Released", "..", "."]:
        p = os.path.join(HERE, c, "twitter_user_data.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="latin-1", low_memory=False)
            return df[df.profile_yn == "yes"] if apply_row_filter else df
    raise FileNotFoundError("twitter_user_data.csv not found")


# ==========================================================================
# Figure 1 - why the count columns were log transformed
# ==========================================================================
def fig_skew():
    raw = load_raw()
    cols = [("tweet_count", "total tweets posted"),
            ("fav_number", "favourites received")]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.22})
    for i, (col, label) in enumerate(cols):
        v = raw[col].dropna()

        ax = axes[i][0]
        ax.hist(v, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
        # Compact tick labels: raw counts run to millions, and the default
        # labels overlap each other at this figure width.
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel(label)
        ax.set_ylabel("number of accounts")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        title(ax, f"{col} — raw",
              f"median {v.median():,.0f}   max {v.max():,.0f}   skewness {v.skew():.2f}")

        ax = axes[i][1]
        lv = np.log1p(v)
        ax.hist(lv, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
        ax.set_xlabel(f"log(1 + {col})")
        ax.set_ylabel("number of accounts")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        title(ax, f"{col} — after log1p", f"skewness {lv.skew():.2f}")

    fig.suptitle("Both count columns collapse into a single bar on their raw scale; "
                 "log1p spreads them out",
                 x=0.005, ha="left", fontsize=12, color=INK, y=1.04)
    fig.savefig(os.path.join(FIG, "fig1_skew_log_transform.png"), **SAVE)
    plt.close(fig)


# ==========================================================================
# Figure 2 - why some columns were dropped
# ==========================================================================
def fig_missing():
    raw = load_raw(apply_row_filter=False)
    miss = (raw.isna().mean() * 100)
    miss = miss[miss > 0].sort_values()

    dropped = {"gender_gold", "profile_yn_gold", "tweet_coord",
               "user_timezone", "tweet_location", "_last_judgment_at"}
    colours = [ORANGE if c in dropped else BLUE for c in miss.index]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(miss.index, miss.values, height=0.62, color=colours, zorder=3)
    for name, v in miss.items():
        ax.annotate(f"{v:.1f}%", (v, name), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.4, color=INK2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("percentage of the 20,050 rows that are missing")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["dropped from the clean table", "kept"],
              frameon=False, fontsize=8.5, loc="lower right")
    title(ax, "Nine of the 26 source columns have missing values",
          "the other 17 are complete; the six in orange are not carried into the clean table")
    fig.savefig(os.path.join(FIG, "fig2_missing_values.png"), **SAVE)
    plt.close(fig)


# ==========================================================================
# Figure 3 - why a random split leaks
# ==========================================================================
def fig_rows_per_account():
    row = pd.read_csv(os.path.join(OUT, "twitter_clean.csv"), low_memory=False)
    per = row.groupby("account_id").size()

    single, multi = int((per == 1).sum()), int((per > 1).sum())
    rows_multi = int(per[per > 1].sum())
    detail = {"2 rows": int((per == 2).sum()),
              "3 rows": int((per == 3).sum()),
              "4-9 rows": int(((per >= 4) & (per <= 9)).sum()),
              "10+ rows": int((per >= 10).sum())}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                 gridspec_kw={"wspace": 0.32,
                                              "width_ratios": [1, 1.1]})

    bars = a1.bar(["exactly 1 row", "more than 1 row"], [single, multi],
                  width=0.5, color=[BLUE, ORANGE], zorder=3)
    for b, v in zip(bars, [single, multi]):
        a1.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color=INK)
    a1.set_ylabel("number of accounts")
    a1.set_ylim(0, single * 1.15)
    a1.grid(axis="y", zorder=0)
    a1.set_axisbelow(True)
    title(a1, "Most accounts appear once",
          f"but {multi} do not, and they cover {rows_multi:,} rows")

    b2 = a2.barh(list(detail.keys())[::-1], list(detail.values())[::-1],
                 height=0.6, color=ORANGE, zorder=3)
    for k, v in zip(list(detail.keys())[::-1], list(detail.values())[::-1]):
        a2.annotate(f"{v:,}", (v, k), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.6, color=INK2)
    a2.set_xlim(0, max(detail.values()) * 1.25)
    a2.set_xlabel("number of accounts")
    a2.grid(axis="x", zorder=0)
    a2.set_axisbelow(True)
    title(a2, "How many rows those accounts have",
          "one account is judged 30 times")

    fig.suptitle("twitter_clean.csv is one row per judged tweet, NOT one row per account",
                 x=0.005, ha="left", fontsize=12, color=INK, y=1.10)
    fig.savefig(os.path.join(FIG, "fig3_rows_per_account.png"), **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    for fn in [fig_skew, fig_missing, fig_rows_per_account]:
        fn()
        print(f"[figure] {fn.__name__}")
    print("\nWritten to output/figures/")
