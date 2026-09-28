"""Shared paths, features and helpers for the clustering scripts.

Works on the account table (one row per account), so accounts judged
several times are not counted several times. Ten structured features form
three views: behaviour, colour and profile. Each feature is z-scored so no
unit dominates Euclidean distance, then divided by sqrt(features in its
view) so every view carries the same weight.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output")
ACCOUNTS_CSV = os.path.join(OUT_DIR, "twitter_clean_accounts.csv")
CLUSTER_DIR = os.path.join(OUT_DIR, "clustering")
TABLE_DIR = os.path.join(CLUSTER_DIR, "tables")
FIG_DIR = os.path.join(CLUSTER_DIR, "figures")
LABELS_CSV = os.path.join(OUT_DIR, "cluster_labels.csv")

SEED = 42

# Twitter's out-of-the-box theme in 2015: link 0084B4, sidebar C0DEED.
DEFAULT_LINK_HEX = "0084B4"
DEFAULT_SIDEBAR_HEX = "C0DEED"

VIEWS = {
    "behaviour": ["log_fav_number", "log_tweet_count", "log_account_age_days"],
    "colour": ["link_color_r", "link_color_g", "link_color_b",
               "sidebar_is_default"],
    "profile": ["has_description", "tweet_has_url", "tweet_has_hashtag"],
}
FEATURES = [col for cols in VIEWS.values() for col in cols]
VIEW_OF = {col: view for view, cols in VIEWS.items() for col in cols}

FEATURE_NOTES = {
    "log_fav_number": "log1p favourites; brands favourite far less (H1)",
    "log_tweet_count": "log1p tweets posted; broadcast volume (H1)",
    "log_account_age_days": "log1p days since creation; age of the account",
    "link_color_r": "link colour red channel, 0-255 (H2)",
    "link_color_g": "link colour green channel, 0-255 (H2)",
    "link_color_b": "link colour blue channel, 0-255 (H2)",
    "sidebar_is_default": "1 when sidebar is the default C0DEED (H2); "
                          "immune to the '0' sidebar defect",
    "has_description": "1 when the profile has a description (completeness)",
    "tweet_has_url": "1 when the sampled tweet carries a link",
    "tweet_has_hashtag": "1 when the sampled tweet carries a hashtag",
}

# Short names for axis labels, in FEATURES order.
FEATURE_LABELS = {
    "log_fav_number": "favourites (log)",
    "log_tweet_count": "tweets (log)",
    "log_account_age_days": "account age (log)",
    "link_color_r": "link red",
    "link_color_g": "link green",
    "link_color_b": "link blue",
    "sidebar_is_default": "default sidebar",
    "has_description": "has description",
    "tweet_has_url": "tweet has URL",
    "tweet_has_hashtag": "tweet has hashtag",
}


def load_accounts() -> pd.DataFrame:
    """Read the account table and add the derived binary columns and label."""
    if not os.path.exists(ACCOUNTS_CSV):
        raise FileNotFoundError(
            f"{ACCOUNTS_CSV} not found - run prepare_data.py first.")
    acct = pd.read_csv(ACCOUNTS_CSV, low_memory=False)
    return acct.assign(
        sidebar_is_default=(acct["sidebar_color_hex"] == DEFAULT_SIDEBAR_HEX)
        .astype(int),
        tweet_has_url=(acct["tweet_n_urls"] > 0).astype(int),
        tweet_has_hashtag=(acct["tweet_n_hashtags"] > 0).astype(int),
        label=acct["is_human"].map({1.0: "human", 0.0: "brand"})
        .fillna("unknown"),
    )


def feature_frame(acct: pd.DataFrame) -> pd.DataFrame:
    """The ten features in original units; missing link colours get the default."""
    default_rgb = {"link_color_r": 0x00, "link_color_g": 0x84,
                   "link_color_b": 0xB4}
    feats = acct[FEATURES].fillna(value=default_rgb)
    if feats.isna().any().any():
        raise ValueError("unexpected missing values in clustering features")
    return feats


def standardise(feats: pd.DataFrame, view_weighted: bool = True):
    """Z-score every feature, optionally equalise the views.

    Returns (Z, mean, sd, weight); `to_original_units` reverses it.
    """
    mean = feats.mean()
    sd = feats.std(ddof=0)
    if (sd == 0).any():
        raise ValueError(f"constant feature(s): {list(sd[sd == 0].index)}")
    weight = pd.Series(1.0, index=feats.columns)
    if view_weighted:
        weight = pd.Series({c: 1.0 / np.sqrt(len(VIEWS[VIEW_OF[c]]))
                            for c in feats.columns})
    Z = ((feats - mean) / sd * weight).to_numpy(dtype=float)
    return Z, mean, sd, weight


def to_original_units(Z_rows: np.ndarray, mean: pd.Series, sd: pd.Series,
                      weight: pd.Series) -> pd.DataFrame:
    """Map standardised rows (centroids, prototypes) back to original units."""
    return pd.DataFrame(Z_rows / weight.to_numpy() * sd.to_numpy()
                        + mean.to_numpy(), columns=mean.index)


def relabel(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Renumber clusters 1..k in ascending order of `score`."""
    order = np.argsort(score)
    lookup = np.empty(len(score), dtype=int)
    lookup[order] = np.arange(1, len(score) + 1)
    return lookup[labels]


def save_table(df: pd.DataFrame, name: str) -> str:
    """Write a table to TABLE_DIR and return its path."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, name)
    df.to_csv(path, index=False)
    return path
