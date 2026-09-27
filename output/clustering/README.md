# Clustering (Task 2) — outputs and how to use them

## Run

```bash
python code/prepare_data.py      # if output/twitter_clean_accounts.csv is missing
python code/task2_cluster.py     # ~2.5 min: models, tables, log, all figures
python code/cluster_figures.py   # optional: redraw figures from the saved tables
```

Code: `code/task2_cluster.py` (models), `code/cluster_common.py` (features,
paths), `code/cluster_figures.py` (figures), `code/myminisom.py` (Lab 3 SOM,
unmodified). Uses only numpy, pandas, scipy, scikit-learn and matplotlib.
Seeded throughout: two consecutive runs give byte-identical tables.

## For Task 4: `output/cluster_labels.csv`

One row per account (18,715), joined on `unit_id` or `account_id`.

| column | meaning |
|---|---|
| `kmeans_cluster` | k-means cluster 1–4 (numbered by brand share, C1 = most human) |
| `som_hier_cluster` | SOM → complete-linkage cluster 1–4 (same numbering rule) |
| `som_node`, `som_x`, `som_y` | best-matching neuron on the 20 × 20 SOM |
| `kmeans_brand_share` | brand share of the account's k-means cluster (confidence-1 labels) |
| `node_n_labelled` | confidence-1 labelled neighbours in the same neuron, **excluding the account itself** |
| `node_brand_share` | their brand share (leave-one-out); NaN when fewer than 10 |
| `cluster_vote` | `brand` / `human` / `abstain`: **the clustering view's vote** |
| `disagree` | 1 when the vote contradicts a known label |
| `suggestion` | `keep`, `amend human->brand`, `amend brand->human`, `first label: …`, `no vote` |

**Vote rule.** A neuron votes brand when its brand share is at least twice the
overall rate (0.262 → ≥ 0.525), human when at most half (≤ 0.131), otherwise
abstains. This is lift, as in association rules (W7). A plain majority was
rejected: 74% of labels are human, so almost every neuron is human-majority,
and on confidence-1 accounts only 55% of brands would get a brand vote.
The lift rule gets 82%, against 90% for humans
(`tables/vote_rule_comparison.csv`). 65% of accounts receive a vote.

**Do not amend on this vote alone.** It is one view out of several. Among
confidence-1 accounts it contradicts 10% of human labels and 18% of brand
labels, which are almost certainly correct.

## Main findings (numbers in `clustering_log.txt`)

1. **The structure is weak.** Silhouette stays between 0.17 and 0.23 for every
   k from 2 to 10, and no k reproduces the human/brand label (ARI ≤ 0.08).
   k = 4 was chosen: it has the same silhouette as k = 3 but a higher
   BSS/TSS (0.38 vs 0.29), it is seed-stable, and adding a fifth cluster
   halves the smallest cluster (18% → 8.5%) and makes the result depend on
   the seed.
2. **Brand-enriched clusters exist, but none is brand-majority.** C3 (no
   description, few favourites, young accounts, default sidebar) and C4
   (tweets carrying URLs and hashtags) are 44–46% brand, against 29% overall.
3. **What separates brand-leaning from human-leaning neurons.** This uses
   the method of Lab 3 Code 6–7: compare the feature means of the two groups.
   The groups are the 49 neurons that vote brand (3,786 accounts) and the
   120 that vote human (8,380 accounts). Differences are in SD units:
   - favourites −1.8 (mean log favourites 1.4 against 7.5)
   - tweet has a URL +1.2 (62% against 5%)
   - has a description −0.66 (63% against 89%)
   - tweet has a hashtag +0.64 (30% against 5%)
   - default sidebar +0.44 (60% against 38%)
   - tweet volume only +0.25

   So H2 (brands keep the default theme) gets moderate support. The
   "tweets a lot" half of H1 barely shows.
4. **The vote tracks annotator doubt.** Among brand-labelled accounts the vote
   disagrees with 75% when confidence is below 0.5, 34% at about two-thirds
   agreement, and 18% when all annotators agree (χ², p = 4e-56). The same
   gradient holds, weaker, for human labels (15%, 14%, 10%; p = 2e-5).
5. **Methods agree moderately.** ARI between k-means and SOM → complete is
   0.42. Removing the 3,893 accounts with a `"0"` colour barely changes the
   result (ARI 0.96). Dropping log1p or the view weighting changes it
   substantially (ARI 0.44–0.49), so those two choices matter and are
   justified in `cluster_common.py`.
6. **Linkage.** All four linkages of the W4 lecture were compared on the SOM
   prototypes. Complete was kept for three reasons: it is the Lab 3 default,
   it agrees best with k-means (ARI 0.42, against 0.18–0.24 for the other
   three), and single linkage chains, leaving a cluster of 0.6%.

   Centroid linkage produces 17 inversions: after two clusters merge, their
   new centroid can lie closer to a third cluster than the merge that was
   just made, so a later merge sits lower in the tree. On such a tree
   scipy's `cut_tree` returns only 3 clusters. `cut_into_k` avoids this by
   replaying the merges in order until 4 clusters remain; on monotonic trees
   it gives the same result as `cut_tree`.

   Complete linkage run directly on 3,000 raw accounts puts 83% of them in
   one cluster, which is why the tree is built on the SOM prototypes
   instead.

## Method provenance (for the report)

Every method comes from the W4 slides, Lab 3, or the course-provided
`myminisom.py`, with four exceptions. Each needs a one-sentence definition
in the report:

| Method | Used for | One-sentence definition |
|---|---|---|
| Silhouette coefficient | answering the W4 diagnostic question "are the clusters well separated?" | For each point, (b − a) / max(a, b), where a is its mean distance to its own cluster and b to the nearest other cluster; it ranges from −1 to 1, and values near 0 mean overlapping clusters. |
| Adjusted Rand index (ARI) | seed stability, method agreement, agreement with labels, sensitivity | The share of point pairs that two partitions treat alike (same cluster in both, or different clusters in both), corrected so that random labelling scores 0 and identical partitions score 1. |
| Chi-square test of independence | whether disagreement with the vote depends on annotator confidence | Compares the observed counts in a contingency table with the counts expected if the two variables were independent. |
| Leave-one-out | computing each neuron's brand share | An account's own label is excluded from the share that decides its vote, so it cannot vote for itself. |

## Figures (`figures/`)

| file | question it answers |
|---|---|
| `cl1_kmeans_choose_k.png` | Which k? (elbow, silhouette, seed stability) |
| `cl2_kmeans_profiles.png` | What separates the clusters, and how do the labels mix in each? |
| `cl3_pca_map.png` | Where do brands sit in the feature space? |
| `cl4_som_maps.png` | SOM: U-matrix, hits, brand share, and the three components that track it |
| `cl5_hierarchical.png` | Dendrogram, the clusters on the SOM grid, and the linkage comparison |
| `cl6_agreement.png` | Do the methods agree with each other and with the labels? How robust are they? |
| `cl7_cluster_vote.png` | Does the vote track annotator confidence? What does it suggest? |
