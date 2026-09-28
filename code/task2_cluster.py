"""Cluster the accounts with k-means, a SOM and hierarchical clustering.

Run after prepare_data.py:   python task2_cluster.py 

Labels never form the clusters; they only describe them afterwards. Each
account's SOM neuron then votes human or brand from its neighbours' labels.

Outputs:
  ../output/cluster_labels.csv            one row per account
  ../output/clustering/tables/*.csv       result tables
  ../output/clustering/clustering_log.txt run log
  ../output/clustering/figures/*.png      via cluster_figures.py
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, is_monotonic, linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import cluster_figures
from cluster_common import (CLUSTER_DIR, FEATURE_NOTES, FEATURES, LABELS_CSV,
                            SEED, VIEW_OF, feature_frame,
                            load_accounts, relabel, save_table, standardise,
                            to_original_units)
from myminisom import MiniSom

# --------------------------------------------------------------------------
# Model constants
# --------------------------------------------------------------------------
K_RANGE = range(2, 11)
N_INIT = 10              # restarts per k-means fit; best WSS is kept (W4)
# k = 4. There is no sharp elbow and the silhouette is flat (0.21-0.23) for
# every k from 3 to 10, so no k separates the data clearly. k = 4 matches
# k = 3 on silhouette but explains 8 points more of the variance
# (BSS/TSS 0.38 vs 0.29) and is seed-stable. Going past 4 buys at most
# +0.02 silhouette (k = 9) while the smallest cluster halves (18% -> 8.5%) and
# k = 5 changes with the seed. W4: "if using more clusters does not better
# distinguish the groups, go with fewer clusters". Section B prints the
# evidence on every run.
K_FINAL = 4
STABILITY_SEEDS = (1, 2, 3, 4)
SIL_SAMPLE = 8_000       # silhouette is O(n^2); averaged over SIL_REPEATS
SIL_REPEATS = 3          # fixed subsamples, identical for every k

SOM_GRIDS = (10, 15, 20, 26)   # 26 x 26 ~ 5 * sqrt(n) neurons (MiniSom rule)
SOM_SIZE = 20            # ~47 accounts per neuron: enough labels per neuron
SOM_LR = 0.5             # initial learning rate, decays linearly to 0
SOM_EPOCHS = 10          # iterations = epochs * n_accounts
# sigma = grid / 2 follows Lab 3 (100 x 100 map, sigma 50); it decays to 1.

# the four cluster-distance definitions of the Week 4 lecture
LINKAGES = ("single", "complete", "average", "centroid")
# complete: the Lab 3 default, and W4: "use complete if you want tight,
# distinct boundaries". On these prototypes it also agrees best with k-means
# of the four linkages, while single linkage chains and leaves a cluster of
# under 1% of accounts. Section D prints all four.
SOM_LINKAGE = "complete"
HIER_SAMPLE = 3_000      # direct linkage control; 4.5 million pairs

# Cluster vote (section F)
MIN_NODE_LABELS = 10     # fewer confidence-1 neighbours -> abstain
# A neuron votes only when its brand share differs from the population's by
# a factor of two: brand if lift >= 2, human if lift <= 1/2 (lift = neuron
# brand share / overall brand share, as for association rules in W7). A
# plain majority is not used: with 74% of labels human, almost every neuron
# is human-majority and the vote would contradict half of the brands.
VOTE_LIFT = 2.0
# gender_confidence takes three levels in practice: ~0.34 (one of three
# annotators agree), ~0.67 (two of three) and 1 (all three).
CONF_BINS = [0.0, 0.5, 0.9999, 1.0]
CONF_NAMES = ["<0.5", "0.5-<1", "=1"]

_log_lines = []


def log(msg: str = "") -> None:
    """Print a line and keep it for clustering_log.txt."""
    print(msg)
    _log_lines.append(str(msg))


def banner(text: str) -> None:
    """Log a section heading."""
    log("\n" + "=" * 74)
    log(text)
    log("=" * 74)


def confident_labelled(acct: pd.DataFrame) -> np.ndarray:
    """Mask of accounts with a known label and confidence 1."""
    return (acct["is_human"].notna() & (acct["gender_confidence"] >= 1.0)
            ).to_numpy()


def overall_brand_share(acct: pd.DataFrame) -> float:
    """Brand share among confidence-1 labels, the base rate for the vote."""
    return float((acct.loc[confident_labelled(acct), "is_human"] == 0).mean())


def brand_share(labels: np.ndarray, acct: pd.DataFrame, k: int) -> np.ndarray:
    """Brand share of each cluster 0..k-1, from confidence-1 labels."""
    mask = confident_labelled(acct)
    is_brand = (acct["is_human"] == 0).to_numpy() & mask
    n = np.bincount(labels[mask], minlength=k)
    b = np.bincount(labels[is_brand], minlength=k)
    return b / np.maximum(n, 1)


# --------------------------------------------------------------------------
# A. Features
# --------------------------------------------------------------------------
def describe_features(feats: pd.DataFrame, weight: pd.Series) -> None:
    """Save and log each feature's view, weight, mean and SD."""
    banner("A. FEATURES - 10 structured columns, 3 views, z-scored")
    table = pd.DataFrame({
        "feature": FEATURES,
        "view": [VIEW_OF[c] for c in FEATURES],
        "weight_after_z": [round(float(weight[c]), 3) for c in FEATURES],
        "mean": [round(float(feats[c].mean()), 3) for c in FEATURES],
        "sd": [round(float(feats[c].std(ddof=0)), 3) for c in FEATURES],
        "why": [FEATURE_NOTES[c] for c in FEATURES],
    })
    save_table(table, "cluster_features.csv")
    log(table[["feature", "view", "weight_after_z", "mean", "sd"]]
        .to_string(index=False))
    log(f"\n  accounts: {len(feats):,}")


# --------------------------------------------------------------------------
# B. k-means
# --------------------------------------------------------------------------
def silhouette(Z: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over SIL_REPEATS fixed subsamples."""
    rng = np.random.RandomState(SEED)
    scores = [silhouette_score(Z[idx], labels[idx]) for idx in
              (rng.choice(len(Z), SIL_SAMPLE, replace=False)
               for _ in range(SIL_REPEATS))]
    return float(np.mean(scores))


def kmeans_k_sweep(Z: np.ndarray, acct: pd.DataFrame) -> pd.DataFrame:
    """Fit k-means for every k in K_RANGE and record the metrics for choosing k.

    ARI with the human/brand label is reported only; it does not choose k.
    """
    banner("B. K-MEANS - choosing k (k = 2..10, n_init = 10)")
    tss = float(((Z - Z.mean(axis=0)) ** 2).sum())
    known = acct["is_human"].notna().to_numpy()
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(Z)
        sizes = np.bincount(km.labels_, minlength=k)
        rows.append({
            "k": k,
            "wss": round(float(km.inertia_), 1),
            "bss_over_tss": round(1.0 - km.inertia_ / tss, 4),
            "silhouette": round(silhouette(Z, km.labels_), 4),
            "smallest_cluster_pct": round(100.0 * sizes.min() / len(Z), 1),
            "min_centroid_distance": round(
                float(pdist(km.cluster_centers_).min()), 3),
            "ari_vs_label": round(adjusted_rand_score(
                acct.loc[known, "label"], km.labels_[known]), 3),
        })
        log("  k={k:>2}  WSS={wss:>9,.1f}  BSS/TSS={bss_over_tss:.3f}  "
            "silhouette={silhouette:.3f}  smallest={smallest_cluster_pct:4.1f}%"
            "  closest centroids={min_centroid_distance:.2f}"
            "  ARI vs label={ari_vs_label:.3f}".format(**rows[-1]))
    sweep = pd.DataFrame(rows)
    sweep["wss_drop_pct"] = (-sweep["wss"].pct_change() * 100).round(1)
    save_table(sweep, "kmeans_k_sweep.csv")
    return sweep


def kmeans_stability(Z: np.ndarray) -> pd.DataFrame:
    """ARI between the seed-42 solution and each of STABILITY_SEEDS, per k."""
    rows = []
    for k in K_RANGE:
        ref = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit_predict(Z)
        aris = [adjusted_rand_score(
            ref, KMeans(n_clusters=k, n_init=N_INIT, random_state=s).fit_predict(Z))
            for s in STABILITY_SEEDS]
        rows.append({"k": k, "min_ari": round(min(aris), 3),
                     "mean_ari": round(float(np.mean(aris)), 3)})
    table = pd.DataFrame(rows)
    save_table(table, "kmeans_stability.csv")
    log("\n  seed stability (ARI vs seed 42, seeds 1-4):")
    log("  " + "  ".join(f"k={r.k}:{r.min_ari:.2f}" for r in table.itertuples()))
    return table


def cluster_profile(acct: pd.DataFrame, labels: np.ndarray,
                    method: str) -> pd.DataFrame:
    """Size, label mix and feature summary of each cluster, in original units."""
    mask = confident_labelled(acct)
    rows = []
    for cid in np.unique(labels):
        grp = acct[labels == cid]
        conf = acct[(labels == cid) & mask]
        rows.append({
            "method": method, "cluster": int(cid), "n": len(grp),
            "pct_accounts": round(100.0 * len(grp) / len(acct), 1),
            "pct_human": round(100.0 * (grp["label"] == "human").mean(), 1),
            "pct_brand": round(100.0 * (grp["label"] == "brand").mean(), 1),
            "pct_unknown": round(100.0 * (grp["label"] == "unknown").mean(), 1),
            "brand_share_conf1": round(float((conf["is_human"] == 0).mean()), 3),
            "median_favourites": float(grp["fav_number"].median()),
            "median_tweets": float(grp["tweet_count"].median()),
            "median_age_days": float(grp["account_age_days"].median()),
            "pct_default_link": round(
                100.0 * (grp["link_color_hex"] == "0084B4").mean(), 1),
            "pct_default_sidebar": round(100.0 * grp["sidebar_is_default"].mean(), 1),
            "pct_description": round(100.0 * grp["has_description"].mean(), 1),
            "pct_tweet_url": round(100.0 * grp["tweet_has_url"].mean(), 1),
            "pct_tweet_hashtag": round(100.0 * grp["tweet_has_hashtag"].mean(), 1),
        })
    return pd.DataFrame(rows)


def kmeans_final(Z: np.ndarray, acct: pd.DataFrame, feats: pd.DataFrame):
    """Fit k = K_FINAL, number clusters by brand share and save profiles."""
    km = KMeans(n_clusters=K_FINAL, n_init=N_INIT, random_state=SEED).fit(Z)
    labels = relabel(km.labels_, brand_share(km.labels_, acct, K_FINAL))
    profile = cluster_profile(acct, labels, "kmeans")
    save_table(profile, "kmeans_profile.csv")
    save_table(pd.crosstab(labels, acct["gender"]).reset_index()
               .rename(columns={"row_0": "cluster"}), "kmeans_by_gender.csv")
    # centroid of each cluster in unweighted z units, for the profile heatmap
    z_plain = (feats - feats.mean()) / feats.std(ddof=0)
    centroids = z_plain.groupby(labels).mean().round(3)
    save_table(centroids.reset_index().rename(columns={"index": "cluster"}),
               "kmeans_centroids_z.csv")
    log(f"\n  final k = {K_FINAL}; clusters numbered by brand share (C1 = most human)")
    log(profile[["cluster", "n", "pct_accounts", "pct_human", "pct_brand",
                 "pct_unknown", "brand_share_conf1"]].to_string(index=False))
    return labels


# --------------------------------------------------------------------------
# C. Self-organising map
# --------------------------------------------------------------------------
def train_som(Z: np.ndarray, size: int) -> MiniSom:
    """Train a size x size MiniSom with PCA initialisation."""
    som = MiniSom(size, size, Z.shape[1], sigma=size / 2.0,
                  learning_rate=SOM_LR, neighborhood_function="gaussian",
                  random_seed=SEED)
    som.pca_weights_init(Z)
    som.train_random(Z, SOM_EPOCHS * len(Z))
    return som


def som_quality(som: MiniSom, Z: np.ndarray, size: int) -> dict:
    """Quantisation error, topographic error and neuron occupancy of a SOM."""
    hits = som.activation_response(Z)
    return {"grid": f"{size}x{size}", "neurons": size * size,
            "quantisation_error": round(float(som.quantization_error(Z)), 4),
            "topographic_error": round(float(som.topographic_error(Z)), 4),
            "empty_neuron_pct": round(100.0 * (hits == 0).mean(), 1),
            "median_hits": float(np.median(hits))}


def som_grid_sweep(Z: np.ndarray) -> pd.DataFrame:
    """Train one SOM per grid size in SOM_GRIDS and compare their quality."""
    banner("C. SELF-ORGANISING MAP - grid size (Lab 3 MiniSom)")
    rows = []
    for size in SOM_GRIDS:
        t0 = time.time()
        row = {**som_quality(train_som(Z, size), Z, size),
               "train_seconds": round(time.time() - t0, 1)}
        rows.append(row)
        log("  {grid:>5}  QE={quantisation_error:.3f}  TE={topographic_error:.3f}"
            "  empty={empty_neuron_pct:4.1f}%  median hits={median_hits:5.0f}"
            "  train={train_seconds:.0f}s".format(**row))
    # timing stays in the log only, so the table is identical on every run
    table = pd.DataFrame(rows).drop(columns="train_seconds")
    save_table(table, "som_grid_sweep.csv")
    log(f"  chosen: {SOM_SIZE}x{SOM_SIZE} - QE keeps falling with more neurons,"
        f" but each neuron must hold enough labelled accounts for a vote (F)")
    return table


def som_final(Z: np.ndarray, acct: pd.DataFrame, mean, sd, weight):
    """Train the chosen SOM; return each account's BMU and a neuron table."""
    som = train_som(Z, SOM_SIZE)
    weights = som.get_weights()
    # BMU of every account (Lab 3 Code 13), stored as a flat neuron index
    xy = np.array([som.winner(x) for x in Z])
    bmu = xy[:, 0] * SOM_SIZE + xy[:, 1]
    n_nodes = SOM_SIZE * SOM_SIZE
    mask = confident_labelled(acct)
    is_brand = (acct["is_human"] == 0).to_numpy() & mask
    n_lab = np.bincount(bmu, weights=mask, minlength=n_nodes)
    n_brand = np.bincount(bmu, weights=is_brand, minlength=n_nodes)
    proto_z = weights.reshape(n_nodes, -1)
    x, y = np.divmod(np.arange(n_nodes), SOM_SIZE)
    nodes = pd.concat([
        pd.DataFrame({
            "node": np.arange(n_nodes), "x": x, "y": y,
            "hits": np.bincount(bmu, minlength=n_nodes),
            "n_conf_labelled": n_lab.astype(int),
            "brand_share_conf1": np.where(n_lab > 0, n_brand / np.maximum(n_lab, 1),
                                          np.nan),
            "umatrix": som.distance_map().reshape(n_nodes),
        }),
        to_original_units(proto_z, mean, sd, weight).add_prefix("proto_"),
        pd.DataFrame(proto_z, columns=[f"z_{c}" for c in FEATURES]),
    ], axis=1)
    q = som_quality(som, Z, SOM_SIZE)
    log(f"\n  final SOM {q['grid']}: QE={q['quantisation_error']:.3f}  "
        f"TE={q['topographic_error']:.3f}  empty neurons={q['empty_neuron_pct']}%")
    return bmu, nodes


def som_brand_drivers(acct: pd.DataFrame, feats: pd.DataFrame,
                      bmu: np.ndarray, nodes: pd.DataFrame) -> pd.DataFrame:
    """Feature means of brand-leaning versus human-leaning neurons.

    Follows Lab 3 Code 6-7; the gap is in SD units so binary flags and
    colour channels share one scale.
    """
    base = overall_brand_share(acct)
    share = nodes["brand_share_conf1"].where(
        nodes["n_conf_labelled"] >= MIN_NODE_LABELS).to_numpy()
    node_side = np.select([share >= VOTE_LIFT * base, share <= base / VOTE_LIFT],
                          ["brand", "human"], default="neither")
    side = node_side[bmu]
    means = feats.groupby(side).mean()
    gap = (means.loc["brand"] - means.loc["human"]) / feats.std(ddof=0)
    table = (pd.DataFrame({
        "feature": FEATURES, "view": [VIEW_OF[c] for c in FEATURES],
        "mean_brand_neurons": means.loc["brand", FEATURES].round(3).to_numpy(),
        "mean_human_neurons": means.loc["human", FEATURES].round(3).to_numpy(),
        "diff_in_sd": gap[FEATURES].round(3).to_numpy()})
        .assign(order=lambda d: -d["diff_in_sd"].abs())
        .sort_values("order").drop(columns="order"))
    save_table(table, "som_brand_drivers.csv")
    log(f"\n  brand-leaning neurons: {int((node_side == 'brand').sum())} "
        f"({int((side == 'brand').sum()):,} accounts); human-leaning: "
        f"{int((node_side == 'human').sum())} ({int((side == 'human').sum()):,})")
    log("  feature means, brand-leaning minus human-leaning, in SD units:")
    log(table.to_string(index=False))
    return table


# --------------------------------------------------------------------------
# D. Hierarchical clustering
# --------------------------------------------------------------------------
def cut_into_k(link: np.ndarray, k: int) -> np.ndarray:
    """Cut a linkage tree into k clusters.

    cut_tree miscounts clusters on trees with inversions (centroid linkage),
    so those trees replay their first n-k merges instead.
    """
    if is_monotonic(link):
        return cut_tree(link, n_clusters=k).ravel()
    n = len(link) + 1
    owner = np.arange(n)            # tree node that currently holds each leaf
    for step, (a, b) in enumerate(link[:n - k, :2].astype(int)):
        owner = np.where(np.isin(owner, (a, b)), n + step, owner)
    return np.unique(owner, return_inverse=True)[1]


def prototype_linkages(nodes: pd.DataFrame, bmu: np.ndarray,
                       km_labels: np.ndarray) -> pd.DataFrame:
    """Compare the four linkages on the occupied SOM prototypes, cut at K_FINAL."""
    banner(f"D. HIERARCHICAL - linkage on the SOM prototypes, cut at k={K_FINAL}")
    occupied = nodes[nodes["hits"] > 0]
    P = occupied[[f"z_{c}" for c in FEATURES]].to_numpy()
    rows = []
    for method in LINKAGES:
        link = linkage(P, method=method)
        acc = propagate(cut_into_k(link, K_FINAL),
                        occupied["node"].to_numpy(), bmu, len(nodes))
        sizes = np.bincount(acc, minlength=K_FINAL)
        rows.append({"input": f"{len(P)} SOM prototypes", "linkage": method,
                     "inversions": int((np.diff(link[:, 2]) < 0).sum()),
                     "sizes": "/".join(str(s) for s in sorted(sizes, reverse=True)),
                     "largest_pct": round(100.0 * sizes.max() / len(bmu), 1),
                     "smallest_pct": round(100.0 * sizes.min() / len(bmu), 1),
                     "ari_vs_kmeans": round(adjusted_rand_score(km_labels, acc), 3)})
        log("  {linkage:<9} inversions={inversions:<3} sizes={sizes:<24}"
            "  ARI vs k-means={ari_vs_kmeans:.3f}".format(**rows[-1]))
    return pd.DataFrame(rows)


def propagate(proto_labels: np.ndarray, proto_nodes: np.ndarray,
              bmu: np.ndarray, n_nodes: int) -> np.ndarray:
    """Give every account the cluster of its best-matching neuron."""
    node_label = np.full(n_nodes, -1)
    node_label[proto_nodes] = proto_labels
    return node_label[bmu]


def som_hier_final(nodes: pd.DataFrame, bmu: np.ndarray, acct: pd.DataFrame):
    """Cluster the SOM prototypes with SOM_LINKAGE and map them to accounts."""
    occupied = nodes[nodes["hits"] > 0]
    P = occupied[[f"z_{c}" for c in FEATURES]].to_numpy()
    cut = cut_into_k(linkage(P, method=SOM_LINKAGE), K_FINAL)
    raw = propagate(cut, occupied["node"].to_numpy(), bmu, len(nodes))
    labels = relabel(raw, brand_share(raw, acct, K_FINAL))
    node_cluster = np.full(len(nodes), 0)
    node_cluster[occupied["node"].to_numpy()] = relabel(
        cut, brand_share(raw, acct, K_FINAL))
    profile = cluster_profile(acct, labels, f"som_{SOM_LINKAGE}")
    save_table(profile, "som_hier_profile.csv")
    save_table(pd.crosstab(labels, acct["gender"]).reset_index()
               .rename(columns={"row_0": "cluster"}), "som_hier_by_gender.csv")
    log(f"\n  SOM -> {SOM_LINKAGE} linkage, k = {K_FINAL}:")
    log(profile[["cluster", "n", "pct_accounts", "pct_human", "pct_brand",
                 "pct_unknown", "brand_share_conf1"]].to_string(index=False))
    return labels, nodes.assign(hier_cluster=node_cluster)


def direct_linkage_control(Z: np.ndarray, acct: pd.DataFrame,
                           km_labels: np.ndarray, som_labels: np.ndarray) -> dict:
    """SOM_LINKAGE directly on a label-stratified sample, to check the SOM step."""
    idx = (acct.groupby("label", group_keys=False)
           .apply(lambda g: g.sample(frac=HIER_SAMPLE / len(acct),
                                     random_state=SEED)).index.to_numpy())
    link = linkage(pdist(Z[idx]), method=SOM_LINKAGE)
    direct = cut_into_k(link, K_FINAL)
    sizes = np.bincount(direct, minlength=K_FINAL)
    row = {"input": f"{len(idx)}-account sample", "linkage": SOM_LINKAGE,
           "inversions": int((np.diff(link[:, 2]) < 0).sum()),
           "sizes": "/".join(str(s) for s in sorted(sizes, reverse=True)),
           "largest_pct": round(100.0 * sizes.max() / len(idx), 1),
           "smallest_pct": round(100.0 * sizes.min() / len(idx), 1),
           "ari_vs_kmeans": round(adjusted_rand_score(km_labels[idx], direct), 3),
           "ari_vs_som_hier": round(adjusted_rand_score(som_labels[idx], direct), 3)}
    log(f"\n  control - direct {SOM_LINKAGE} linkage on {len(idx):,} accounts:"
        f" sizes={row['sizes']}  ARI vs k-means={row['ari_vs_kmeans']:.3f}"
        f"  ARI vs SOM->{SOM_LINKAGE}={row['ari_vs_som_hier']:.3f}")
    log(f"  -> on raw accounts the farthest-pair rule isolates a few far-out"
        f" accounts ({row['smallest_pct']}% in the smallest cluster, "
        f"{row['largest_pct']}% in the largest); prototypes average them away")
    return row


# --------------------------------------------------------------------------
# E. Agreement and sensitivity
# --------------------------------------------------------------------------
def agreement(acct: pd.DataFrame, km: np.ndarray, sh: np.ndarray) -> pd.DataFrame:
    """ARI between the two clusterings and the two label codings.

    Pairs involving a label use labelled accounts only.
    """
    banner("E. AGREEMENT (adjusted Rand index) AND SENSITIVITY")
    known = acct["is_human"].notna().to_numpy()
    parts = {"k-means": km, f"SOM->{SOM_LINKAGE}": sh,
             "human/brand label": acct["label"].to_numpy(),
             "gender label": acct["gender"].to_numpy()}
    names = list(parts)
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            use = known if "label" in a + b else np.ones(len(acct), bool)
            mat.loc[a, b] = mat.loc[b, a] = adjusted_rand_score(
                parts[a][use], parts[b][use])
    save_table(mat.round(4).reset_index().rename(columns={"index": "method"}),
               "agreement_ari.csv")
    log(mat.round(3).to_string())
    return mat


def sensitivity(acct: pd.DataFrame, feats: pd.DataFrame,
                km_labels: np.ndarray) -> pd.DataFrame:
    """Refit k-means under three alternative preprocessing choices."""
    zero = ((acct["link_color_zero_suspect"] == 1) |
            (acct["sidebar_color_zero_suspect"] == 1)).to_numpy()
    # same column names, raw counts in place of log1p (names kept so the
    # view weighting still applies)
    raw_counts = feats.assign(log_fav_number=acct["fav_number"],
                              log_tweet_count=acct["tweet_count"],
                              log_account_age_days=acct["account_age_days"])
    variants = [
        ("views not equalised (plain z-score)", feats, True, False),
        ("raw counts instead of log1p", raw_counts, True, True),
        ("'0' colour accounts removed", feats, ~zero, True),
    ]
    rows = []
    for name, data, keep, weighted in variants:
        keep = np.ones(len(feats), bool) if keep is True else keep
        Z, *_ = standardise(data[keep], view_weighted=weighted)
        lab = KMeans(n_clusters=K_FINAL, n_init=N_INIT,
                     random_state=SEED).fit_predict(Z)
        rows.append({"variant": name, "accounts": int(keep.sum()),
                     "silhouette": round(silhouette(Z, lab), 3),
                     "smallest_cluster_pct": round(
                         100.0 * np.bincount(lab).min() / keep.sum(), 1),
                     "ari_vs_main": round(adjusted_rand_score(km_labels[keep], lab), 3)})
        log("  {variant:<38} n={accounts:>6,}  silhouette={silhouette:.3f}  "
            "smallest={smallest_cluster_pct:4.1f}%  ARI vs main={ari_vs_main:.3f}"
            .format(**rows[-1]))
    table = pd.DataFrame(rows)
    save_table(table, "sensitivity.csv")
    return table


# --------------------------------------------------------------------------
# F. Cluster vote
# --------------------------------------------------------------------------
def neuron_share(acct: pd.DataFrame, bmu: np.ndarray, n_nodes: int):
    """Leave-one-out brand share of each account's SOM neuron.

    Uses confidence-1 labels; NaN when fewer than MIN_NODE_LABELS remain.
    """
    mask = confident_labelled(acct).astype(float)
    is_brand = ((acct["is_human"] == 0).to_numpy() * mask)
    n_loo = np.bincount(bmu, weights=mask, minlength=n_nodes)[bmu] - mask
    b_loo = np.bincount(bmu, weights=is_brand, minlength=n_nodes)[bmu] - is_brand
    share = np.where(n_loo >= MIN_NODE_LABELS, b_loo / np.maximum(n_loo, 1), np.nan)
    return n_loo.astype(int), share


def apply_vote(share: np.ndarray, brand_min: float, human_max: float):
    """brand / human / abstain; NaN shares (too few labels) abstain."""
    valid = ~np.isnan(share)
    safe = np.where(valid, share, 0.5)
    return np.select([valid & (safe >= brand_min), valid & (safe <= human_max)],
                     ["brand", "human"], default="abstain")


def compare_vote_rules(acct: pd.DataFrame, share: np.ndarray,
                       base: float) -> pd.DataFrame:
    """Coverage and per-class agreement of three vote rules on confidence-1 labels."""
    rules = [("plain majority", 0.5, 0.5),
             ("majority, abstain 0.35-0.65", 0.65, 0.35),
             (f"lift >= {VOTE_LIFT:g} / <= 1/{VOTE_LIFT:g} (chosen)",
              VOTE_LIFT * base, base / VOTE_LIFT)]
    conf = confident_labelled(acct)
    lab = acct["label"].to_numpy()
    rows = []
    for name, brand_min, human_max in rules:
        vote = apply_vote(share, brand_min, human_max)
        row = {"rule": name, "brand_if_share_ge": round(brand_min, 3),
               "human_if_share_le": round(human_max, 3),
               "coverage_pct": round(100.0 * (vote != "abstain").mean(), 1)}
        for cls in ["human", "brand"]:
            voted = conf & (lab == cls) & (vote != "abstain")
            row[f"conf1_{cls}_agree_pct"] = round(
                100.0 * (vote[voted] == cls).mean(), 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    save_table(table, "vote_rule_comparison.csv")
    log(f"  overall brand share (confidence-1 labels) = {base:.3f}")
    log(table.to_string(index=False))
    return table


def suggestion(label: pd.Series, vote: np.ndarray) -> np.ndarray:
    """Combine label and vote into keep / amend / first label / no vote."""
    lab = label.to_numpy()
    return np.select(
        [vote == "abstain",
         lab == "unknown",
         vote == lab],
        ["no vote", np.char.add("first label: ", vote.astype(str)), "keep"],
        default=np.char.add(np.char.add("amend ", lab.astype(str)),
                            np.char.add("->", vote.astype(str))))


def build_label_table(acct, km, sh, bmu, km_share) -> pd.DataFrame:
    """Cast the neuron vote and assemble the per-account output table."""
    banner("F. CLUSTER VOTE - SOM neuron, leave-one-out, lift rule")
    n_loo, share = neuron_share(acct, bmu, SOM_SIZE * SOM_SIZE)
    base = overall_brand_share(acct)
    compare_vote_rules(acct, share, base)
    vote = apply_vote(share, VOTE_LIFT * base, base / VOTE_LIFT)
    sug = suggestion(acct["label"], vote)
    x, y = np.divmod(bmu, SOM_SIZE)
    return pd.DataFrame({
        "unit_id": acct["unit_id"], "account_id": acct["account_id"],
        "gender": acct["gender"], "gender_confidence": acct["gender_confidence"],
        "is_human": acct["is_human"],
        "label_conflict_human_brand": acct["label_conflict_human_brand"],
        "kmeans_cluster": km, "som_hier_cluster": sh,
        "som_node": bmu, "som_x": x, "som_y": y,
        "kmeans_brand_share": np.round(km_share[km - 1], 3),
        "node_n_labelled": n_loo, "node_brand_share": np.round(share, 3),
        "cluster_vote": vote,
        "disagree": pd.Series(sug).str.startswith("amend").astype(int).to_numpy(),
        "suggestion": sug,
    })


def validate_vote(labels: pd.DataFrame) -> pd.DataFrame:
    """Disagreement rate by annotator confidence, per label, with a chi-square test."""
    log("\n  validation against annotator confidence:")
    known = labels[labels["is_human"].notna()].assign(
        conf_bin=lambda d: pd.cut(d["gender_confidence"], CONF_BINS,
                                  labels=CONF_NAMES, include_lowest=True))
    voted = known[known["cluster_vote"] != "abstain"]
    rows = []
    for (lab, cbin), grp in voted.groupby([voted["is_human"].map(
            {1.0: "human", 0.0: "brand"}), "conf_bin"], observed=True):
        rows.append({"label": lab, "confidence": cbin, "voted": len(grp),
                     "disagree": int(grp["disagree"].sum()),
                     "disagree_pct": round(100.0 * grp["disagree"].mean(), 1)})
    table = pd.DataFrame(rows)
    for lab in ["human", "brand"]:
        sub = voted[voted["is_human"] == (1.0 if lab == "human" else 0.0)]
        chi2, p, dof, _ = chi2_contingency(pd.crosstab(sub["conf_bin"],
                                                       sub["disagree"]))
        log(f"  {lab}-labelled: chi-square(disagree x confidence bin) = "
            f"{chi2:.1f}, dof={dof}, p={p:.2g}")
    save_table(table, "vote_validation.csv")
    log(table.to_string(index=False))
    return table


def summarise_votes(labels: pd.DataFrame) -> None:
    """Log vote coverage and agreement; save suggestion counts."""
    known = labels["is_human"].notna()
    conf1 = known & (labels["gender_confidence"] >= 1.0)
    voted = conf1 & (labels["cluster_vote"] != "abstain")
    log(f"\n  coverage: {100 * (labels.cluster_vote != 'abstain').mean():.1f}% "
        f"of accounts receive a vote")
    log(f"  confidence-1 accounts, leave-one-out: vote agrees with label for "
        f"{100 * (1 - labels.loc[voted, 'disagree'].mean()):.1f}% of "
        f"{int(voted.sum()):,} voted")
    conflict = labels["label_conflict_human_brand"] == 1
    log(f"  accounts labelled both human and brand across rows: "
        f"{int(conflict.sum())}, of which the vote disagrees with the kept "
        f"label for {int(labels.loc[conflict, 'disagree'].sum())}")
    summary = (labels.assign(conf_lt_1=labels["gender_confidence"] < 1.0)
               .groupby(["suggestion", "conf_lt_1"]).size()
               .unstack(fill_value=0)
               .rename(columns={False: "confidence_eq_1",
                                True: "confidence_lt_1"})
               .reset_index())
    save_table(summary, "vote_suggestions.csv")
    log("\n" + summary.to_string(index=False))


# --------------------------------------------------------------------------
def main() -> None:
    """Run every step, draw the figures and write the log."""
    t0 = time.time()
    acct = load_accounts()
    feats = feature_frame(acct)
    Z, mean, sd, weight = standardise(feats)
    describe_features(feats, weight)

    kmeans_k_sweep(Z, acct)
    kmeans_stability(Z)
    km = kmeans_final(Z, acct, feats)

    som_grid_sweep(Z)
    bmu, nodes = som_final(Z, acct, mean, sd, weight)
    som_brand_drivers(acct, feats, bmu, nodes)

    linkage_table = prototype_linkages(nodes, bmu, km)
    sh, nodes = som_hier_final(nodes, bmu, acct)
    control = direct_linkage_control(Z, acct, km, sh)
    save_table(pd.concat([linkage_table, pd.DataFrame([control])],
                         ignore_index=True), "hier_linkage_comparison.csv")
    save_table(nodes, "som_nodes.csv")

    agreement(acct, km, sh)
    sensitivity(acct, feats, km)

    km_share = brand_share(km - 1, acct, K_FINAL)
    labels = build_label_table(acct, km, sh, bmu, km_share)
    labels.to_csv(LABELS_CSV, index=False)
    validate_vote(labels)
    summarise_votes(labels)

    banner("FIGURES")
    for path in cluster_figures.make_all():
        log(f"  {os.path.relpath(path, CLUSTER_DIR)}")
    log(f"\n  total run time {time.time() - t0:.0f}s")
    with open(os.path.join(CLUSTER_DIR, "clustering_log.txt"), "w") as f:
        f.write("\n".join(_log_lines) + "\n")


if __name__ == "__main__":
    main()
