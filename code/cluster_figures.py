"""Figures cl1-cl7 for the clustering results.

Reads only the tables and cluster_labels.csv that task2_cluster.py writes,
so the figures can be redrawn without refitting:  python cluster_figures.py
Style follows make_figures.py; human is blue and brand orange throughout.
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from scipy.cluster.hierarchy import dendrogram, linkage

from cluster_common import (FEATURE_LABELS, FEATURES, FIG_DIR, LABELS_CSV,
                            TABLE_DIR, VIEWS, feature_frame, load_accounts,
                            standardise)

# --- palette (identical to make_figures.py) --------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"        # human, brand
UNKNOWN_GREY = "#b9b8b0"
CLUSTER_COLOURS = ["#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#e6f0fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_GREY = LinearSegmentedColormap.from_list(
    "seq_grey", ["#f4f3f0", "#c3c2b7", "#898781", "#3b3a37"])
# brand share: blue (all human) - grey (0.5) - orange (all brand)
DIV_BRAND = LinearSegmentedColormap.from_list(
    "div_brand", ["#184f95", "#6da7ec", "#f0efec", "#f3a47f", "#b84a1d"])
# centroid z-scores: blue (below mean) - grey - red (above mean)
DIV_Z = LinearSegmentedColormap.from_list(
    "div_z", ["#184f95", "#6da7ec", "#f0efec", "#ef8d8c", "#a92e2d"])
for cmap in (SEQ_BLUE, SEQ_GREY, DIV_BRAND, DIV_Z):
    cmap.set_bad(SURFACE)

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
MIN_DISPLAY_LABELS = 5    # SOM neurons with fewer labelled accounts: blank


def title(ax, text, sub=None):
    """Title and subtitle above an axes, offset in points."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 24 if sub else 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=10.5, color=INK,
                fontweight="medium", annotation_clip=False)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.3, color=INK2,
                    annotation_clip=False)


def suptitle(fig, text, y=1.08):
    """Left-aligned figure title."""
    fig.suptitle(text, x=0.005, ha="left", fontsize=12, color=INK, y=y)


def table(name):
    """Read a saved result table."""
    return pd.read_csv(os.path.join(TABLE_DIR, name))


def save(fig, name):
    """Save a figure to FIG_DIR, close it and return its path."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, **SAVE)
    plt.close(fig)
    return path


def small_colorbar(fig, im, ax, label):
    """Narrow colorbar without outline, in muted ink."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    cb.set_label(label, fontsize=8, color=INK2)
    return cb


# ==========================================================================
# cl1 - choosing k
# ==========================================================================
def _k_panel(ax, k, values, k_final, ylabel, head, sub, ylim=None):
    """One metric against k, with the chosen k marked."""
    ax.plot(k, values, color=BLUE, lw=2, marker="o", ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    chosen = values[list(k).index(k_final)]
    ax.plot([k_final], [chosen], marker="o", ms=9, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.annotate(f"k = {k_final}", (k_final, chosen), textcoords="offset points",
                xytext=(8, 8), fontsize=8.5, color=INK)
    ax.set_xticks(list(k))
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    title(ax, head, sub)


def fig_choose_k():
    """cl1: elbow, silhouette and seed stability against k."""
    sweep = table("kmeans_k_sweep.csv")
    stab = table("kmeans_stability.csv")
    k_final = len(table("kmeans_profile.csv"))
    k = sweep["k"].to_numpy()
    after = sweep.loc[sweep.k == k_final + 1, "wss_drop_pct"].iloc[0]
    unstable = stab.loc[stab.min_ari < 0.9, "k"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5),
                             gridspec_kw={"wspace": 0.34})
    _k_panel(axes[0], k, sweep["wss"].to_numpy() / 1000, k_final,
             "within-cluster sum of squares (thousands)", "Elbow: WSS",
             f"no sharp bend; adding cluster {k_final + 1} removes only "
             f"{after:.0f}%")
    _k_panel(axes[1], k, sweep["silhouette"].to_numpy(), k_final,
             "mean silhouette", "Silhouette",
             "0 to 0.25 means overlapping clusters", ylim=(0, 0.3))
    _k_panel(axes[2], k, stab["min_ari"].to_numpy(), k_final,
             "worst ARI vs seed 42 (4 other seeds)", "Seed stability",
             "k = " + ", ".join(map(str, unstable)) + " change with the seed"
             if unstable else "every k is seed-stable", ylim=(0, 1.05))
    suptitle(fig, f"k-means: k = {k_final} is chosen on WSS and stability; "
                  "no k produces well-separated clusters", y=1.12)
    return save(fig, "cl1_kmeans_choose_k.png")


# ==========================================================================
# cl2 - cluster profiles
# ==========================================================================
def _profile_heatmap(fig, ax, cent, prof):
    """k-means centroids in SD units, one column per feature, grouped by view."""
    M = cent[FEATURES].to_numpy()
    im = ax.imshow(M, cmap=DIV_Z, vmin=-1.3, vmax=1.3, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.8,
                color=SURFACE if abs(v) > 0.85 else INK)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in FEATURES], rotation=35,
                       ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"C{c}   n={n:,}" for c, n in zip(prof.cluster, prof.n)],
                       fontsize=8.5, color=INK)
    edge = 0
    for view, cols in VIEWS.items():
        ax.text(edge + (len(cols) - 1) / 2, -0.75, view, ha="center",
                va="bottom", fontsize=8.5, color=INK2)
        edge += len(cols)
        if edge < len(FEATURES):
            ax.axvline(edge - 0.5, color=SURFACE, lw=4)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    small_colorbar(fig, im, ax, "cluster mean (SD from overall mean)")


def _label_mix(ax, prof):
    """Stacked human / brand / unknown shares of each cluster."""
    y = np.arange(len(prof))
    left = np.zeros(len(prof))
    for col, colour, name in [("pct_human", BLUE, "human"),
                              ("pct_brand", ORANGE, "brand"),
                              ("pct_unknown", UNKNOWN_GREY, "unknown")]:
        ax.barh(y, prof[col], left=left, height=0.62, color=colour,
                edgecolor=SURFACE, linewidth=1.5, label=name, zorder=3)
        left = left + prof[col].to_numpy()
    for yi, (h, b) in enumerate(zip(prof.pct_human, prof.pct_brand)):
        ax.text(h + b / 2, yi, f"{b:.0f}%", ha="center", va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of accounts in the cluster")
    ax.invert_yaxis()
    ax.grid(axis="x", zorder=0)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower left",
              bbox_to_anchor=(0, 1.0), handlelength=1, columnspacing=1)


def fig_profiles():
    """cl2: k-means centroids and the label mix of each cluster."""
    cent = table("kmeans_centroids_z.csv")
    prof = table("kmeans_profile.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.4),
                                 gridspec_kw={"width_ratios": [2.4, 1],
                                              "wspace": 0.12})
    _profile_heatmap(fig, a1, cent, prof)
    _label_mix(a2, prof)
    overall = (prof.pct_brand * prof.n).sum() / prof.n.sum()
    rich = prof[prof.pct_brand > 1.3 * overall]
    names = " and ".join(f"C{c}" for c in rich.cluster)
    # No shares in the title: the bars count all labels, while the report
    # quotes confidence-1 shares, so any number here would contradict one.
    suptitle(fig, f"k-means profiles: {names} are brand-enriched, "
                  f"but no cluster is brand-majority",
             y=1.1)
    return save(fig, "cl2_kmeans_profiles.png")


# ==========================================================================
# cl3 - PCA projection
# ==========================================================================
def _pc_label(loadings, var, i):
    """Axis label naming the two features that load most on PC i+1."""
    top = np.argsort(-np.abs(loadings[i]))[:2]
    parts = [f"{FEATURE_LABELS[FEATURES[j]]} {'+' if loadings[i, j] > 0 else '−'}"
             for j in top]
    return f"PC{i + 1} ({100 * var[i]:.0f}% of variance): " + ", ".join(parts)


def _mark_centroids(ax, P, km):
    """Mark and name each k-means cluster mean on the PCA plane."""
    for c in np.unique(km):
        cx, cy = P[km == c].mean(axis=0)
        ax.plot(cx, cy, "o", ms=8, color=INK, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.annotate(f"C{c}", (cx, cy), textcoords="offset points",
                    xytext=(7, 5), fontsize=9, color=INK, fontweight="medium")


def fig_pca_map():
    """cl3: account density and brand share on the first two PCs."""
    acct = load_accounts()
    Z, *_ = standardise(feature_frame(acct))
    lab = pd.read_csv(LABELS_CSV).set_index("unit_id").loc[acct["unit_id"]]
    Zc = Z - Z.mean(axis=0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    var = S ** 2 / (S ** 2).sum()
    km = lab["kmeans_cluster"].to_numpy()
    conf = (lab["is_human"].notna() & (lab["gender_confidence"] >= 1)).to_numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True,
                                 sharey=True, gridspec_kw={"wspace": 0.12})
    hb = a1.hexbin(P[:, 0], P[:, 1], gridsize=42, bins="log", mincnt=1,
                   cmap=SEQ_BLUE, linewidths=0.2, edgecolors=SURFACE)
    small_colorbar(fig, hb, a1, "accounts per cell (log scale)")
    title(a1, "Density of all 18,715 accounts",
          "dots: k-means centroids projected onto the same plane")
    brand = (lab["is_human"] == 0).to_numpy().astype(float)
    hb2 = a2.hexbin(P[conf, 0], P[conf, 1], C=brand[conf], gridsize=42,
                    reduce_C_function=np.mean, mincnt=8, cmap=DIV_BRAND,
                    vmin=0, vmax=1, linewidths=0.2, edgecolors=SURFACE)
    small_colorbar(fig, hb2, a2, "brand share (confidence-1 labels)")
    title(a2, "Brand share per cell",
          "grey = half brand; cells with < 8 labelled accounts left blank")
    for ax in (a1, a2):
        _mark_centroids(ax, P, km)
        ax.set_xlabel(_pc_label(Vt, var, 0), fontsize=8)
    a1.set_ylabel(_pc_label(Vt, var, 1), fontsize=8)
    suptitle(fig, "First two principal components: brand-heavy regions exist, "
                  "but they overlap human regions rather than forming "
                  "separate islands", y=1.1)
    return save(fig, "cl3_pca_map.png")


# ==========================================================================
# cl4 - SOM maps
# ==========================================================================
def _grid(nodes, col):
    """Reshape one neuron column into a 2-D array for imshow."""
    size = int(np.sqrt(len(nodes)))
    arr = np.full((size, size), np.nan)
    arr[nodes["x"], nodes["y"]] = nodes[col]
    return arr.T          # imshow rows = y, columns = x, origin lower


def _som_panel(fig, ax, arr, cmap, head, sub, cbar, vmin=None, vmax=None,
               norm=None):
    """Draw one SOM map with its colorbar and title."""
    im = ax.imshow(arr, origin="lower", cmap=cmap, interpolation="none",
                   **({"norm": norm} if norm else {"vmin": vmin, "vmax": vmax}))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    small_colorbar(fig, im, ax, cbar)
    title(ax, head, sub)


def _component_panel(driver):
    """Panel spec for one component plane, from a som_brand_drivers row."""
    col = driver.feature
    binary = col in ("sidebar_is_default", "has_description",
                     "tweet_has_url", "tweet_has_hashtag")
    if binary:
        cbar, lo, hi = "share of accounts with it", 0, 1
    elif col.startswith("log_"):
        cbar, lo, hi = f"log(1 + {col[4:].replace('_', ' ')})", None, None
    else:
        cbar, lo, hi = "channel value, 0-255", 0, 255
    return (f"proto_{col}", SEQ_BLUE, f"Component: {FEATURE_LABELS[col]}",
            f"brand- minus human-leaning neurons: {driver.diff_in_sd:+.2f} SD",
            cbar, lo, hi)


def fig_som_maps():
    """cl4: U-matrix, hits, brand share and the top three component planes."""
    nodes = table("som_nodes.csv")
    drivers = table("som_brand_drivers.csv").head(3)
    shown = nodes.assign(
        hits_nan=nodes["hits"].where(nodes["hits"] > 0),
        share_shown=nodes["brand_share_conf1"].where(
            nodes["n_conf_labelled"] >= MIN_DISPLAY_LABELS))
    size = int(np.sqrt(len(nodes)))
    panels = [
        ("umatrix", SEQ_GREY, "U-matrix", "dark = large jump to neighbours",
         "mean neighbour distance (scaled)", None, None),
        ("hits_nan", SEQ_BLUE, "Hits", f"{int((nodes.hits == 0).sum())} empty "
         "neurons left blank", "accounts per neuron (log scale)", None, None),
        ("share_shown", DIV_BRAND, "Brand share",
         f"blank: < {MIN_DISPLAY_LABELS} confidence-1 labels",
         "brand share", 0, 1),
    ] + [_component_panel(d) for d in drivers.itertuples()]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    for ax, (col, cmap, head, sub, cbar, lo, hi) in zip(axes.flat, panels):
        norm = LogNorm(vmin=1, vmax=shown[col].max()) if col == "hits_nan" else None
        _som_panel(fig, ax, _grid(shown, col), cmap, head, sub, cbar, lo, hi, norm)
    suptitle(fig, f"Self-organising map ({size} x {size}): brand share (top "
                  "right) and the three features that best separate brand- "
                  "from human-leaning neurons (bottom row)", y=1.0)
    return save(fig, "cl4_som_maps.png")


# ==========================================================================
# cl5 - hierarchical clustering
# ==========================================================================
def _dendrogram(ax, nodes, k):
    """Complete-linkage dendrogram of the occupied prototypes, cut at k."""
    occupied = nodes[nodes["hits"] > 0]
    link = linkage(occupied[[f"z_{c}" for c in FEATURES]].to_numpy(),
                   method="complete")
    heights = np.sort(link[:, 2])
    cut = 0.5 * (heights[-k] + heights[-(k - 1)])
    dendrogram(link, ax=ax, truncate_mode="lastp", p=30, no_labels=True,
               color_threshold=0, above_threshold_color=INK2)
    ax.axhline(cut, color=INK, lw=1.2)
    ax.annotate(f"cut: {k} clusters", (1, cut), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(-4, 4), ha="right",
                fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    ax.set_ylabel("complete-linkage merge distance")
    ax.set_xlabel(f"{len(occupied)} occupied SOM prototypes (last 30 merges shown)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    title(ax, "Dendrogram of the SOM prototypes",
          "complete linkage: cluster distance = farthest pair")


def _hier_on_grid(ax, nodes, k):
    """Hierarchical clusters coloured on the SOM grid."""
    size = int(np.sqrt(len(nodes)))
    arr = _grid(nodes.assign(c=nodes["hier_cluster"].where(nodes["hits"] > 0)), "c")
    cmap = ListedColormap(CLUSTER_COLOURS[:k])
    cmap.set_bad(SURFACE)
    ax.imshow(arr, origin="lower", cmap=cmap, vmin=0.5, vmax=k + 0.5,
              interpolation="none")
    # thin surface lines between neurons: the gap is the secondary encoding
    for g in np.arange(-0.5, size, 1):
        ax.axhline(g, color=SURFACE, lw=0.8)
        ax.axvline(g, color=SURFACE, lw=0.8)
    for c in range(1, k + 1):
        cells = nodes[(nodes.hier_cluster == c) & (nodes.hits > 0)]
        cx, cy = cells[["x", "y"]].mean()
        near = cells.iloc[np.argmin(np.hypot(cells.x - cx, cells.y - cy))]
        ax.text(near.x, near.y, f"C{c}", ha="center", va="center", fontsize=9,
                color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    title(ax, "The same clusters on the SOM grid",
          "C1 = lowest brand share; blank = empty neuron")


def _linkage_bars(ax, comp):
    """Cluster sizes and ARI with k-means for each linkage."""
    ramp = ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6"]
    names = [f"{r.linkage}\n({'sample' if 'sample' in r.input else 'SOM'})"
             for r in comp.itertuples()]
    y = np.arange(len(comp))
    for i, r in enumerate(comp.itertuples()):
        sizes = np.array([int(s) for s in r.sizes.split("/")], float)
        pct = 100 * sizes / sizes.sum()
        left = np.concatenate([[0], np.cumsum(pct)[:-1]])
        ax.barh([i] * len(pct), pct, left=left, height=0.6, color=ramp[:len(pct)],
                edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(pct[0] / 2, i, f"{pct[0]:.0f}%", ha="center", va="center",
                fontsize=8, color=SURFACE)
        ax.text(103, i, f"ARI {r.ari_vs_kmeans:.2f}", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("cluster sizes, largest first (% of the accounts clustered)")
    ax.grid(axis="x", zorder=0)
    som_rows = comp.loc[comp.input.str.contains("SOM")]
    best = som_rows.sort_values("ari_vs_kmeans").iloc[-1]
    smallest = som_rows.sort_values("smallest_pct").iloc[0]
    title(ax, "Linkage choice", f"right: ARI with k-means, highest for "
          f"{best.linkage}; {smallest.linkage} leaves a "
          f"{smallest.smallest_pct:.1f}% cluster")


def fig_hierarchical():
    """cl5: dendrogram, clusters on the SOM grid and linkage comparison."""
    nodes = table("som_nodes.csv")
    comp = table("hier_linkage_comparison.csv")
    k = int(nodes["hier_cluster"].max())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0),
                             gridspec_kw={"width_ratios": [1.35, 0.9, 1.1],
                                          "wspace": 0.3})
    _dendrogram(axes[0], nodes, k)
    _hier_on_grid(axes[1], nodes, k)
    _linkage_bars(axes[2], comp)
    suptitle(fig, "Hierarchical clustering: SOM prototypes are merged with "
                  "complete linkage (Lab 3) and cut into four clusters", y=1.12)
    return save(fig, "cl5_hierarchical.png")


# ==========================================================================
# cl6 - agreement and sensitivity
# ==========================================================================
def fig_agreement():
    """cl6: ARI between partitions and sensitivity to preprocessing."""
    ari = table("agreement_ari.csv").set_index("method")
    sens = table("sensitivity.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.7),
                                 gridspec_kw={"width_ratios": [1, 1.1],
                                              "wspace": 0.75})
    im = a1.imshow(ari.to_numpy(), cmap=SEQ_BLUE, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(ari.to_numpy()):
        a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color=SURFACE if v > 0.55 else INK)
    a1.set_xticks(range(len(ari)))
    a1.set_xticklabels(ari.columns, rotation=30, ha="right", fontsize=8)
    a1.set_yticks(range(len(ari)))
    a1.set_yticklabels(ari.index, fontsize=8)
    a1.tick_params(length=0)
    for side in a1.spines.values():
        side.set_visible(False)
    small_colorbar(fig, im, a1, "adjusted Rand index")
    title(a1, "Agreement between partitions",
          "1 = identical, 0 = no better than chance")

    a2.barh([textwrap.fill(v, 20) for v in sens["variant"]], sens["ari_vs_main"],
            height=0.55, color=BLUE, zorder=3)
    a2.tick_params(axis="y", labelsize=8.5)
    for yi, v in enumerate(sens["ari_vs_main"]):
        a2.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    a2.set_xlim(0, 1.1)
    a2.invert_yaxis()
    a2.set_xlabel("ARI with the main k-means solution")
    a2.grid(axis="x", zorder=0)
    title(a2, "Sensitivity to preprocessing", "k-means refitted with one choice changed")
    suptitle(fig, "The clusterings agree with each other far more than "
                  "with the human/brand label", y=1.12)
    return save(fig, "cl6_agreement.png")


# ==========================================================================
# cl7 - cluster vote validation
# ==========================================================================
def _disagreement_bars(ax, vv):
    """Share of accounts the vote contradicts, by label and confidence."""
    order = ["<0.5", "0.5-<1", "=1"]
    x = np.arange(len(order))
    for off, lab, colour in [(-0.2, "human", BLUE), (0.2, "brand", ORANGE)]:
        sub = vv[vv["label"] == lab].set_index("confidence").reindex(order)
        ax.bar(x + off, sub["disagree_pct"], width=0.38, color=colour,
               label=f"labelled {lab}", zorder=3)
        for xi, (v, n) in enumerate(zip(sub["disagree_pct"], sub["voted"])):
            if not np.isnan(v):
                ax.annotate(f"{v:.0f}%", (xi + off, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=7.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["< 0.5\n(about 1 in 3 agree)", "0.5 – <1\n(about 2 in 3)",
                        "= 1\n(all agree)"])
    ax.set_xlabel("annotator confidence of the label")
    ax.set_ylabel("% whose SOM neuron votes the other way")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    title(ax, "Disagreement falls as annotators agree more",
          "accounts that received a vote; confidence-1 scored leave-one-out")


def _suggestion_bars(ax, sug):
    """Count of each suggestion, split by label confidence."""
    order = ["amend human->brand", "amend brand->human",
             "first label: human", "first label: brand"]
    s = sug.set_index("suggestion").reindex(order).fillna(0)
    y = np.arange(len(order))
    ax.barh(y, s["confidence_lt_1"], height=0.6, color=INK2,
            label="label confidence < 1", zorder=3)
    ax.barh(y, s["confidence_eq_1"], left=s["confidence_lt_1"], height=0.6,
            color=AXIS, edgecolor=SURFACE, linewidth=1.5,
            label="label confidence = 1", zorder=3)
    for yi, total in enumerate(s["confidence_lt_1"] + s["confidence_eq_1"]):
        ax.text(total + 20, yi, f"{int(total):,}", va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([o.replace("->", " to ") for o in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("accounts")
    ax.set_xlim(0, (s.sum(axis=1).max()) * 1.18)
    ax.grid(axis="x", zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    title(ax, "What the cluster view suggests",
          "one vote among several in Task 4 - not an amendment on its own")


def fig_vote():
    """cl7: vote disagreement by confidence and suggestion counts."""
    vv = table("vote_validation.csv")
    sug = table("vote_suggestions.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9),
                                 gridspec_kw={"width_ratios": [1, 1.1],
                                              "wspace": 0.42})
    _disagreement_bars(a1, vv)
    _suggestion_bars(a2, sug)
    suptitle(fig, "Cluster vote: low-confidence labels contradict their "
                  "neighbourhood more often", y=1.12)
    return save(fig, "cl7_cluster_vote.png")


def make_all():
    """Draw every figure and return their paths."""
    return [fn() for fn in (fig_choose_k, fig_profiles, fig_pca_map,
                            fig_som_maps, fig_hierarchical, fig_agreement,
                            fig_vote)]


if __name__ == "__main__":
    for path in make_all():
        print(f"[figure] {os.path.relpath(path, os.path.dirname(FIG_DIR))}")
