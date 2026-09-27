"""
CSCI946 Assignment 2 - Classification
======================================



Run:
    python classification.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures, do not open windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------------------
SEED = 42            # fixed seed so every result is reproducible
HIGH_CONF = 1.0      # a label is "trusted" only if all annotators agreed
FLAG_PROB = 0.80     # models must be at least this sure the label is wrong

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_NAME = "twitter_clean_accounts.csv"
OUT = os.path.join(HERE, "classification_output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def find_data():
    """Look for the data file next to this script or in the usual folders."""
    candidates = [os.path.join(HERE, DATA_NAME),
                  os.path.join(HERE, "A2_2026_Released", DATA_NAME),
                  os.path.join(HERE, "output", DATA_NAME),
                  os.path.join(HERE, "..", "output", DATA_NAME)]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Cannot find " + DATA_NAME + ". Looked in:\n  "
                            + "\n  ".join(os.path.abspath(p) for p in candidates))


# Numeric features (behaviour + colour + text statistics).
# gender, gender_confidence, name and label_* are NEVER used: they are the label.
NUMERIC = ["log_fav_number", "log_tweet_count", "log_account_age_days",
           "tweets_per_day", "favs_per_day", "has_retweets",
           "link_color_r", "link_color_g", "link_color_b",
           "link_color_brightness", "link_color_zero_suspect",
           "sidebar_color_r", "sidebar_color_g", "sidebar_color_b",
           "sidebar_color_brightness", "sidebar_color_zero_suspect",
           "has_description", "description_len", "description_word_count",
           "description_n_urls", "description_n_mentions",
           "description_n_hashtags", "tweet_len", "tweet_word_count",
           "tweet_n_urls", "tweet_n_mentions", "tweet_n_hashtags"]
TEXT = ["description_clean", "tweet_clean"]


def banner(msg):
    print("\n" + "=" * 60 + "\n" + msg + "\n" + "=" * 60)


# ---------------------------------------------------------------------------
# 1. Load data and split
# ---------------------------------------------------------------------------
banner("1. LOAD DATA")
df = pd.read_csv(find_data())
df[TEXT] = df[TEXT].fillna("")
df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median())   # 2 unusable colours
for c in ["tweets_per_day", "favs_per_day"]:             # very skewed rates
    df[c] = np.log1p(df[c])

labelled = df[df["is_human"].notna()].copy()
labelled["is_human"] = labelled["is_human"].astype(int)
trusted = labelled[labelled["gender_confidence"] >= HIGH_CONF]

# 70/30 stratified split on trusted labels (one row per account -> no leakage)
train, test = train_test_split(trusted, test_size=0.30,
                               stratify=trusted["is_human"], random_state=SEED)
y_train, y_test = train["is_human"].values, test["is_human"].values
print(f"labelled accounts: {len(labelled):,}  trusted: {len(trusted):,}  "
      f"(train {len(train):,} / test {len(test):,})")


# ---------------------------------------------------------------------------
# 2. The three models
# ---------------------------------------------------------------------------
def tfidf():
    """TF-IDF for one text column: words and 2-word phrases, English stop
    words removed, rare terms (<3 accounts) ignored."""
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3,
                           stop_words="english", sublinear_tf=True)


def build_models():
    text_only = ColumnTransformer([("desc", tfidf(), "description_clean"),
                                   ("tweet", tfidf(), "tweet_clean")])
    all_feats = lambda: ColumnTransformer(
        [("num", StandardScaler(), NUMERIC),
         ("desc", tfidf(), "description_clean"),
         ("tweet", tfidf(), "tweet_clean")])
    return {
        # Naive Bayes needs non-negative counts/weights -> text only.
        # alpha = Laplace smoothing (lecture) so unseen words never give P=0.
        "Naive Bayes": Pipeline([("pre", text_only),
                                 ("clf", MultinomialNB(alpha=0.5))]),
        # class_weight="balanced" corrects the ~74% human / 26% brand imbalance
        "Logistic Regression": Pipeline([("pre", all_feats()),
                                         ("clf", LogisticRegression(
                                             max_iter=2000,
                                             class_weight="balanced"))]),
        # LinearSVC gives no probabilities -> calibrate to get them for ROC
        "Linear SVM": Pipeline([("pre", all_feats()),
                                ("clf", CalibratedClassifierCV(
                                    LinearSVC(C=0.1, class_weight="balanced",
                                              max_iter=5000), cv=3))]),
    }


# ---------------------------------------------------------------------------
# 3. Train and evaluate on the held-out test set
# ---------------------------------------------------------------------------
banner("2. TEST-SET RESULTS")
models = build_models()
rows, probs = [], {}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, m) in zip(axes, models.items()):
    m.fit(train, y_train)
    p = m.predict_proba(test)[:, 1]              # P(human)
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    rows.append({"model": name,
                 "accuracy": accuracy_score(y_test, pred),
                 "precision": precision_score(y_test, pred, average="macro"),
                 "recall": recall_score(y_test, pred, average="macro"),
                 "f1": f1_score(y_test, pred, average="macro"),
                 "roc_auc": roc_auc_score(y_test, p)})
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["brand", "human"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "c1_confusion_matrices.png"), dpi=150)
plt.close()

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)
print(metrics.to_string(index=False))

# ROC curves
plt.figure(figsize=(6, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves (test set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "c2_roc_curves.png"), dpi=150)
plt.close()

# Words that push the logistic regression towards brand or human
lr = models["Logistic Regression"]
coef = pd.Series(lr.named_steps["clf"].coef_[0],
                 index=lr.named_steps["pre"].get_feature_names_out())
coef = coef[~coef.index.str.startswith("num__")]
top = pd.concat([coef.nsmallest(15), coef.nlargest(15)])
top.index = top.index.str.replace("desc__", "bio: ").str.replace("tweet__", "tweet: ")
plt.figure(figsize=(8, 8))
plt.barh(top.index, top.values, color=["tab:red"] * 15 + ["tab:blue"] * 15)
plt.axvline(0, c="k", lw=0.8)
plt.title("Words pointing to brand (red) vs human (blue)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "c3_top_words.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 4. Mislabel detection
# ---------------------------------------------------------------------------
# Every labelled account is scored by models that never saw it (5-fold,
# out-of-fold). Models are trained on trusted labels only. A label is
# flagged when the three models on average are >= 80% sure it is wrong
# and none of them supports it.
banner("3. MISLABEL DETECTION")
names = list(models)
oof = pd.DataFrame(index=labelled.index, columns=names, dtype=float)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, te_idx in folds.split(labelled, labelled["is_human"]):
    tr = labelled.iloc[tr_idx]
    tr = tr[tr["gender_confidence"] >= HIGH_CONF]     # train on trusted only
    te = labelled.iloc[te_idx]
    for name, m in build_models().items():
        oof.loc[te.index, name] = m.fit(tr, tr["is_human"]).predict_proba(te)[:, 1]

is_h = np.repeat((labelled["is_human"].values == 1)[:, None], len(names), axis=1)
p_label = oof.where(is_h, 1 - oof)                    # P(recorded label is right)
labelled["p_human"] = oof.mean(axis=1)
labelled["recorded_label"] = np.where(labelled["is_human"] == 1, "human", "brand")
labelled["suggested_label"] = np.where(labelled["p_human"] >= 0.5, "human", "brand")
labelled["flagged"] = ((p_label.mean(axis=1) <= 1 - FLAG_PROB) &
                       (p_label.max(axis=1) < 0.5)).astype(int)

flagged = labelled[labelled["flagged"] == 1]
print(f"flagged: {len(flagged):,} of {len(labelled):,} labelled accounts "
      f"({len(flagged)/len(labelled):.1%})")
print(f"  recorded human, looks brand : {(flagged['recorded_label']=='human').sum()}")
print(f"  recorded brand, looks human : {(flagged['recorded_label']=='brand').sum()}")

# Checks that the flags are meaningful
print(f"mean annotator confidence: flagged {flagged['gender_confidence'].mean():.3f}"
      f" vs others {labelled.loc[labelled['flagged']==0,'gender_confidence'].mean():.3f}")
conflict = labelled["label_conflict_human_brand"] == 1
print(f"flag rate: accounts annotators labelled both human AND brand "
      f"{labelled.loc[conflict,'flagged'].mean():.1%} vs others "
      f"{labelled.loc[~conflict,'flagged'].mean():.1%}")

flagged.sort_values("p_human")[
    ["account_id", "name", "recorded_label", "suggested_label",
     "gender_confidence", "p_human", "description_raw", "tweet_raw"]
].round(4).to_csv(os.path.join(OUT, "flagged_mislabels.csv"), index=False)

# Figure: model belief vs recorded label
plt.figure(figsize=(7, 4.5))
for lab, col in [("human", "tab:blue"), ("brand", "tab:red")]:
    plt.hist(labelled.loc[labelled["recorded_label"] == lab, "p_human"],
             bins=40, alpha=0.6, color=col, label=f"recorded {lab}")
plt.axvline(0.5, c="k", ls="--")
plt.xlabel("average out-of-fold P(human) of the three models")
plt.ylabel("accounts")
plt.title("Model belief vs recorded label (overlap = candidate mislabels)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "c4_mislabel_detection.png"), dpi=150)
plt.close()

banner("DONE - results in classification_output/")