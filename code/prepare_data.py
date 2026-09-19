"""
CSCI946 Assignment 2 - Data Preparation
Author: Tianyu He (9495204)

Turns the raw CrowdFlower Twitter file into modelling-ready tables for the
rest of the group.

Run:  python prepare_data.py

Outputs (written to ../output/):
    twitter_clean.csv           one row per JUDGED TWEET (19,953 rows)
    twitter_clean_accounts.csv  one row per ACCOUNT      (18,715 rows)
    label_conflicts.csv         accounts whose rows carry contradictory labels
    column_guide.csv            every output column, its type and its meaning
    data_quality_report.txt     full audit trail of this run

IMPORTANT - THE RAW FILE IS NOT ONE ROW PER ACCOUNT
    791 accounts appear on more than one row (2,029 rows in total; one account
    appears 30 times). Each row is an (account, sampled tweet) pair that was
    judged independently, so the same account can carry different labels on
    different rows. A random train/test split over the row-level table
    therefore leaks: the same account lands on both sides and every score is
    optimistic. Use `account_id` with GroupShuffleSplit / GroupKFold, or use
    twitter_clean_accounts.csv, which is already one row per account.

    `account_id` is `name` + `created`. This was verified: across all 791
    repeated groups, link_color, sidebar_color, tweet_location and
    user_timezone are identical within every group, so the rows really are
    the same account rather than a name collision.
"""

import os
import re
import glob
import html
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


# ==========================================================================
# 0. Locate input and output
# ==========================================================================
def _find_raw(here):
    for candidate in [os.path.join(here, "..", "A2_2026_Released"),
                      os.path.join(here, "A2_2026_Released"),
                      here, os.path.join(here, "..")]:
        hit = glob.glob(os.path.join(candidate, "twitter_user_data.csv"))
        if hit:
            return hit[0]
    raise FileNotFoundError(
        "twitter_user_data.csv not found. Place it in a folder named "
        "A2_2026_Released beside this script, or in the same folder.")


RAW_PATH = _find_raw(HERE)
OUT_DIR = os.path.join(HERE, "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


# ==========================================================================
# 1. Colour repair
# ==========================================================================
HEX6 = re.compile(r"^[0-9A-F]{6}$")
HEX_SHORT = re.compile(r"^[0-9A-F]{1,6}$")


def repair_colour(series):
    """Return (clean_hex, status) for a raw Twitter colour column.

    Three defects are present in the raw file and each is repaired or flagged
    separately rather than being lumped together:

    * scientific notation - a spreadsheet read a hex code such as 221E07 as
      the number 221 x 10^7 and displayed it as 2.21E+09. Writing the three
      significant digits back in front of 'E' multiplies the mantissa by 100,
      so the exponent must be reduced by 2: 2.21E+09 -> 221E07. This is
      verified numerically in verify_data.py: float(repaired) == float(raw)
      for all 40 repaired values. Caveat: when the mantissa ends in 0 (for
      example 2.00E+69) a reading with leading zeros (020E68, 002E69) gives
      the same number; the no-leading-zero reading is chosen, affecting 8
      values. One value, 9.00E+00, has an exponent below 2 and therefore no
      six-character 'dddEdd' reading, so it is left invalid rather than
      guessed.
    * stripped leading zeros - the same spreadsheet turned 009999 into 9999
      (592 rows). Values of 1-5 hex characters are left-padded with zeros.
    * the literal value "0" - 4,006 sidebar_color rows (20.1%) and 417
      link_color rows. Padding gives 000000 (black), but nothing in the data
      says whether that is a real black theme or an export artefact. "0" is
      the second most common sidebar value, ahead of white, which is
      suspicious. These are padded but flagged as `zero_suspect` so the group
      can exclude them; they are NOT silently treated as ordinary colours.

    status values: 'ok' | 'padded' | 'sci_repaired' | 'zero_suspect' | 'invalid'
    """
    s = series.astype(str).str.strip().str.upper()
    clean = pd.Series(index=s.index, dtype=object)
    status = pd.Series("invalid", index=s.index, dtype=object)

    is_zero = s == "0"
    clean[is_zero], status[is_zero] = "000000", "zero_suspect"

    # 'd.ddE+XX' -> 'ddd' + 'E' + (XX - 2). Moving the three significant
    # digits in front of 'E' multiplies the mantissa by 100, so the exponent
    # must drop by 2 for the repaired hex to denote the same number.
    parts = s.str.extract(r"^([0-9])\.([0-9]{2})E\+([0-9]{2})$")
    is_sci = parts[2].notna() & ~is_zero & (parts[2].fillna("0").astype(int) >= 2)
    exp = (parts.loc[is_sci, 2].astype(int) - 2).astype(str).str.zfill(2)
    clean[is_sci] = parts.loc[is_sci, 0] + parts.loc[is_sci, 1] + "E" + exp
    status[is_sci] = "sci_repaired"

    is_ok = s.str.match(HEX6) & ~is_zero & ~is_sci
    clean[is_ok], status[is_ok] = s[is_ok], "ok"

    is_short = s.str.match(HEX_SHORT) & ~is_zero & ~is_sci & ~is_ok
    clean[is_short], status[is_short] = s[is_short].str.zfill(6), "padded"

    # anything still not six valid hex characters is unusable
    bad = ~clean.fillna("").str.match(HEX6)
    clean[bad], status[bad] = np.nan, "invalid"
    return clean, status


def hex_to_rgb(clean_hex):
    def part(h, i):
        return int(h[i:i + 2], 16) if isinstance(h, str) else np.nan
    r = clean_hex.apply(lambda h: part(h, 0))
    g = clean_hex.apply(lambda h: part(h, 2))
    b = clean_hex.apply(lambda h: part(h, 4))
    return r, g, b


# ==========================================================================
# 2. Text cleaning
# ==========================================================================
# The source contains malformed URLs as well as well-formed ones:
#   'http//www.bnntv.org'      - the colon was lost
#   'http:https://t.co/xxxx'   - two prefixes concatenated
# so the colon is optional and a leftover bare 'http'/'https' token is
# removed afterwards.
# No \b before the scheme: the source glues URLs onto the preceding token
# ('WEBhttp://...', 'bookhttp://...') and onto mojibake ('Brink\xc3\xa5\xc3\x8ahttps://...'),
# and there is no word boundary between a letter and 'h'.
URL_RE = re.compile(r"(?:https?:?//|www\.)\S*", re.IGNORECASE)
LEFTOVER_URL_RE = re.compile(r"\bhttps?\b:?", re.IGNORECASE)
# (?<!\w) stops e-mail addresses such as info@site.com being counted as
# mentions and turned into "info MENTIONTOKEN .com".
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"#\w+")
# Underscores glued to a corrupted run on either side ('_\xd9\xf7\xe2', '\x89\xdb_')
# are part of the corruption, so they are removed with the non-ASCII bytes.
NON_ASCII_RE = re.compile(r"_*(?:[^\x00-\x7F]_*)+")
WS_RE = re.compile(r"\s+")


def unescape_all(s, max_rounds=5):
    """Repeatedly unescape HTML entities until the string stops changing.

    The source contains double and triple escaping ('J&amp;amp;J',
    '&amp;amp;amp;'), so a single html.unescape leaves an entity behind.
    """
    for _ in range(max_rounds):
        new = html.unescape(s)
        if new == s:
            return new
        s = new
    return s


def clean_text(s):
    """Normalise one tweet or profile description.

    The raw CSV is not valid UTF-8 (first invalid byte at offset 927): emoji
    were already corrupted before the file was distributed and survive as
    mojibake runs such as '_Ù÷â'. Left in place they inflate every length
    feature and become tokens in any bag-of-words model, so they are removed.
    HTML entities (&amp;, &lt;, ...) are unescaped repeatedly because the
    source is multiply escaped, and URLs, @mentions and
    #hashtags are replaced by single placeholder tokens so that their
    PRESENCE survives tokenisation while the specific handle or link - which
    is unique per row and therefore useless to a model - does not.
    """
    if not isinstance(s, str):
        return ""
    s = unescape_all(s)
    s = URL_RE.sub(" URLTOKEN ", s)
    s = LEFTOVER_URL_RE.sub(" URLTOKEN ", s)
    s = MENTION_RE.sub(" MENTIONTOKEN ", s)
    s = HASHTAG_RE.sub(" HASHTAGTOKEN ", s)
    s = NON_ASCII_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()


def text_features(raw_series, prefix):
    raw = raw_series.fillna("").astype(str)
    cleaned = raw.apply(clean_text)
    return pd.DataFrame({
        f"{prefix}_raw": raw,
        f"{prefix}_clean": cleaned,
        f"has_{prefix}": (raw.str.strip() != "").astype(int),
        f"{prefix}_len": cleaned.str.len(),
        f"{prefix}_word_count": cleaned.str.split().str.len().fillna(0).astype(int),
        f"{prefix}_n_urls": raw.apply(lambda t: len(URL_RE.findall(t))),
        f"{prefix}_n_mentions": raw.apply(lambda t: len(MENTION_RE.findall(t))),
        f"{prefix}_n_hashtags": raw.apply(lambda t: len(HASHTAG_RE.findall(t))),
        f"{prefix}_n_mojibake": raw.apply(lambda t: len(NON_ASCII_RE.findall(t))),
    })


# ==========================================================================
# 3. Main pipeline
# ==========================================================================
def main():
    log("=" * 74)
    log("CSCI946 A2 - DATA PREPARATION")
    log("=" * 74)

    # ---- 3.1 load ------------------------------------------------------
    # latin-1, not utf-8: the file contains bytes that are not valid UTF-8.
    df = pd.read_csv(RAW_PATH, encoding="latin-1", low_memory=False)
    log(f"\n[1/8 LOAD] {os.path.relpath(RAW_PATH, HERE)}")
    log(f"          {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ---- 3.2 row filter -------------------------------------------------
    n0 = len(df)
    unreadable = df["profile_yn"] == "no"
    assert df.loc[unreadable, "gender"].isna().all(), \
        "profile_yn=='no' rows were expected to have no gender label"
    df = df[~unreadable].copy().reset_index(drop=True)
    log(f"\n[2/8 ROW FILTER]")
    log(f"          removed {int(unreadable.sum())} rows with profile_yn == 'no'")
    log(f"          (verified: all of them have a missing gender, so no label is lost)")
    log(f"          exact duplicate rows: {int(df.duplicated().sum())}")
    log(f"          {n0:,} -> {len(df):,} rows")

    out = pd.DataFrame(index=df.index)

    # ---- 3.3 account identity ------------------------------------------
    out["unit_id"] = df["_unit_id"]
    out["name"] = df["name"].astype(str)
    out["account_id"] = df["name"].astype(str) + "|" + df["created"].astype(str)
    out["n_rows_for_account"] = out.groupby("account_id")["unit_id"].transform("size")

    n_accounts = out["account_id"].nunique()
    multi = out[out["n_rows_for_account"] > 1]
    log(f"\n[3/8 ACCOUNT IDENTITY]")
    log(f"          distinct accounts           : {n_accounts:,}")
    log(f"          accounts with >1 row        : {multi['account_id'].nunique():,}")
    log(f"          rows belonging to those     : {len(multi):,}")
    log(f"          largest account row count   : {int(out['n_rows_for_account'].max())}")
    # verify the key: account-level attributes must not vary inside a group
    for col in ["link_color", "sidebar_color", "tweet_location", "user_timezone"]:
        varies = df.groupby(out["account_id"])[col].nunique(dropna=False).gt(1).sum()
        log(f"          {col:<15} varies within account: {int(varies)}")
        assert varies == 0, f"{col} varies within an account - account_id is unsafe"

    # ---- 3.4 labels -----------------------------------------------------
    out["gender"] = df["gender"]
    out["gender_confidence"] = df["gender:confidence"]
    out["is_human"] = df["gender"].map({"female": 1, "male": 1, "brand": 0})
    out["label_known"] = out["is_human"].notna().astype(int)

    conflict_hb = (out.dropna(subset=["is_human"])
                      .groupby("account_id")["is_human"].nunique() > 1)
    conflict_gender = out.groupby("account_id")["gender"].nunique(dropna=True) > 1
    out["label_conflict_human_brand"] = (
        out["account_id"].map(conflict_hb).astype("boolean").fillna(False).astype(int))
    out["label_conflict_gender"] = (
        out["account_id"].map(conflict_gender).astype("boolean").fillna(False).astype(int))

    log(f"\n[4/8 LABELS]")
    log(f"          human (female+male)         : {int((out.is_human == 1).sum()):,}")
    log(f"          non-human (brand)           : {int((out.is_human == 0).sum()):,}")
    log(f"          unknown (kept, is_human=NaN): {int(out.is_human.isna().sum()):,}")
    log(f"          rows with confidence < 1.0  : {int((out.gender_confidence < 1).sum()):,}")
    log(f"          accounts labelled BOTH human and non-human : "
        f"{int(conflict_hb.sum())}")
    log(f"          accounts with any gender disagreement      : "
        f"{int(conflict_gender.sum())}")
    log(f"          -> these are direct evidence of the assignment's target and are")
    log(f"             exported to label_conflicts.csv rather than silently merged.")

    # ---- 3.5 behaviour counts -------------------------------------------
    log(f"\n[5/8 BEHAVIOUR COUNTS]")
    for col in ["fav_number", "retweet_count", "tweet_count"]:
        out[col] = df[col]
        out["log_" + col] = np.log1p(df[col])
        log(f"          {col:<14} median {df[col].median():>10,.0f}   "
            f"max {df[col].max():>12,.0f}   skew {df[col].skew():>8.2f}")
    out["has_retweets"] = (df["retweet_count"] > 0).astype(int)
    log(f"          retweet_count is 0 for {(df.retweet_count == 0).mean() * 100:.1f}% of rows;")
    log(f"          has_retweets is provided because the raw count is near-constant.")

    # ---- 3.6 dates -------------------------------------------------------
    created = pd.to_datetime(df["created"], format="%m/%d/%y %H:%M", errors="coerce")
    captured = pd.to_datetime(df["tweet_created"], format="%m/%d/%y %H:%M", errors="coerce")
    out["account_age_days"] = (captured - created).dt.days
    out["log_account_age_days"] = np.log1p(out["account_age_days"].clip(lower=0))
    out["tweets_per_day"] = df["tweet_count"] / out["account_age_days"].clip(lower=1)
    out["favs_per_day"] = df["fav_number"] / out["account_age_days"].clip(lower=1)
    log(f"\n[6/8 DATES]")
    log(f"          created parsed             : {int(created.notna().sum()):,} / {len(df):,}")
    log(f"          created range              : {created.min():%Y-%m-%d} .. {created.max():%Y-%m-%d}")
    log(f"          capture window             : {captured.min():%Y-%m-%d %H:%M} .. {captured.max():%Y-%m-%d %H:%M}")
    log(f"          account_age_days range     : {out.account_age_days.min():,.0f} .. {out.account_age_days.max():,.0f}")

    # ---- 3.7 colours ------------------------------------------------------
    log(f"\n[7/8 COLOURS]")
    for col in ["link_color", "sidebar_color"]:
        clean_hex, status = repair_colour(df[col])
        r, g, b = hex_to_rgb(clean_hex)
        out[f"{col}_hex"] = clean_hex
        out[f"{col}_r"], out[f"{col}_g"], out[f"{col}_b"] = r, g, b
        out[f"{col}_brightness"] = 0.299 * r + 0.587 * g + 0.114 * b
        out[f"{col}_status"] = status
        out[f"{col}_usable"] = (status != "invalid").astype(int)
        out[f"{col}_zero_suspect"] = (status == "zero_suspect").astype(int)
        counts = status.value_counts().to_dict()
        log(f"          {col}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))

    # ---- 3.8 text ---------------------------------------------------------
    out = pd.concat([out,
                     text_features(df["description"], "description"),
                     text_features(df["text"], "tweet")], axis=1)
    log(f"\n[8/8 TEXT]")
    for p in ["description", "tweet"]:
        log(f"          {p:<12} non-empty {int(out['has_' + p].sum()):,}   "
            f"rows with mojibake {int((out[p + '_n_mojibake'] > 0).sum()):,}   "
            f"rows with URL {int((out[p + '_n_urls'] > 0).sum()):,}")
    log(f"          raw text kept as *_raw; *_clean is unescaped, URL/@/# replaced")
    log(f"          by tokens, corrupted non-ASCII removed. Length and word-count")
    log(f"          features are computed on the CLEANED text.")

    # ---- 3.9 validation ---------------------------------------------------
    log(f"\n[VALIDATION]")
    checks = [
        ("no duplicate unit_id", out.unit_id.duplicated().sum() == 0),
        ("is_human in {0,1,NaN}", set(out.is_human.dropna().unique()) <= {0.0, 1.0}),
        ("gender_confidence in [0,1]",
         bool(out.gender_confidence.dropna().between(0, 1).all())),
        ("RGB within 0-255",
         bool(all(out[c].dropna().between(0, 255).all()
                  for c in out.columns if c.endswith(("_r", "_g", "_b"))))),
        ("account_age_days >= 0", bool((out.account_age_days.dropna() >= 0).all())),
        ("no NaN in engineered count columns",
         not out[["log_fav_number", "log_tweet_count", "tweets_per_day"]].isna().any().any()),
        ("text columns have no NaN",
         not out[[c for c in out.columns if c.endswith(("_raw", "_clean"))]].isna().any().any()),
        ("row count unchanged since filter", len(out) == len(df)),
    ]
    for label, ok in checks:
        log(f"          [{'PASS' if ok else 'FAIL'}] {label}")
        assert ok, f"validation failed: {label}"

    # ---- 3.10 account-level table -----------------------------------------
    # One row per account, for models that must not see the same account twice.
    # Aggregation rules:
    #   counts   -> max across the account's rows (they are snapshots taken at
    #               slightly different times; tweet_count varies within 205 of
    #               the 791 repeated accounts)
    #   label    -> the label from the row with the highest gender_confidence
    #   text     -> the tweet from that same row, plus a tweet count
    #   colours, dates, ratios -> constant within an account, take the first
    order = out.sort_values(["account_id", "gender_confidence"],
                            ascending=[True, False])
    # drop_duplicates keeps ONE WHOLE ROW per account. groupby().first() must
    # not be used here: it returns the first non-null value column by column,
    # so an account whose top row is 'unknown' (is_human NaN) would be given
    # is_human from a different row, producing records such as
    # gender='unknown' with is_human=1.
    acct = order.drop_duplicates("account_id", keep="first").reset_index(drop=True)
    count_max = out.groupby("account_id")[["fav_number", "retweet_count", "tweet_count"]].max()
    for c in ["fav_number", "retweet_count", "tweet_count"]:
        acct[c] = acct["account_id"].map(count_max[c])
        acct["log_" + c] = np.log1p(acct[c])
    acct["tweets_per_day"] = acct["tweet_count"] / acct["account_age_days"].clip(lower=1)
    acct["favs_per_day"] = acct["fav_number"] / acct["account_age_days"].clip(lower=1)
    acct["has_retweets"] = (acct["retweet_count"] > 0).astype(int)

    # ---- 3.11 conflicts ----------------------------------------------------
    conf_ids = out.loc[out.label_conflict_gender == 1, "account_id"].unique()
    conflicts = (out[out.account_id.isin(conf_ids)]
                 [["account_id", "name", "unit_id", "gender", "gender_confidence",
                   "is_human", "label_conflict_human_brand", "tweet_raw"]]
                 .sort_values(["label_conflict_human_brand", "account_id"],
                              ascending=[False, True]))

    # ---- 3.12 column guide -------------------------------------------------
    descriptions = {
        "unit_id": "CrowdFlower row id. Unique per row, NOT per account.",
        "name": "Twitter screen name. Identifier only - never use as a feature.",
        "account_id": "name + created. The true account key. Split on this.",
        "n_rows_for_account": "How many rows this account has in the file.",
        "gender": "Raw annotator label: female / male / brand / unknown. LABEL - never a feature.",
        "gender_confidence": "Annotator agreement, 0-1. Low = candidate mislabel. "
                             "Label metadata - never a feature (leaks the label).",
        "is_human": "1 = female or male, 0 = brand, NaN = unknown. The target.",
        "label_known": "1 when is_human is not missing. Filter only - never a feature.",
        "label_conflict_human_brand": "1 when this account is labelled both human and brand. "
                                      "Derived from labels - never a feature.",
        "label_conflict_gender": "1 when this account's rows disagree on gender at all. "
                                 "Derived from labels - never a feature.",
        "fav_number": "Favourites received. Raw, heavily right-skewed.",
        "retweet_count": "Retweets of the sampled tweet. 0 for about 97% of rows.",
        "tweet_count": "Total tweets posted. Raw, heavily right-skewed.",
        "has_retweets": "1 when retweet_count > 0. Use instead of the raw count.",
        "account_age_days": "Days between account creation and the capture date.",
        "tweets_per_day": "tweet_count / account_age_days. A rate, not a total.",
        "favs_per_day": "fav_number / account_age_days.",
        "link_color_hex": "Repaired six-digit hex, or NaN if unusable.",
        "link_color_status": "ok / padded / sci_repaired / zero_suspect / invalid.",
        "link_color_usable": "0 when the colour could not be repaired.",
        "link_color_zero_suspect": "1 when the raw value was the literal '0'. Its RGB is "
                                   "0,0,0 but may not be real black - exclude or test both ways.",
        "description_raw": "Profile description exactly as in the source file.",
        "description_clean": "Unescaped, URL/@/# tokenised, mojibake removed.",
        "tweet_raw": "Sampled tweet exactly as in the source file.",
        "tweet_clean": "Cleaned tweet. Use this for TF-IDF / bag of words.",
    }
    guide_rows = []
    for c in out.columns:
        desc = descriptions.get(c)
        if desc is None:
            base = c.replace("sidebar_color", "link_color")
            desc = descriptions.get(base)
        if desc is None and c.startswith("log_"):
            desc = f"log1p of {c[4:]}. Use for distance-based models."
        if desc is None and c.endswith(("_r", "_g", "_b")):
            desc = "Red / green / blue component, 0-255."
        if desc is None and c.endswith("_brightness"):
            desc = "0.299R + 0.587G + 0.114B. Perceived lightness, 0-255."
        if desc is None and c.endswith("_n_urls"):
            desc = "Number of URLs in the raw text."
        if desc is None and c.endswith("_n_mentions"):
            desc = "Number of @mentions in the raw text."
        if desc is None and c.endswith("_n_hashtags"):
            desc = "Number of #hashtags in the raw text."
        if desc is None and c.endswith("_n_mojibake"):
            desc = "Number of corrupted non-ASCII runs found in the raw text."
        if desc is None and c.endswith("_len"):
            desc = "Character length of the CLEANED text."
        if desc is None and c.endswith("_word_count"):
            desc = "Word count of the CLEANED text."
        if desc is None and c.startswith("has_"):
            desc = f"1 when {c[4:]} is present and non-empty."
        guide_rows.append({
            "column": c,
            "dtype": str(out[c].dtype),
            "missing": int(out[c].isna().sum()),
            "n_unique": int(out[c].nunique(dropna=True)),
            "description": desc or "(see script)",
        })
    guide = pd.DataFrame(guide_rows)
    assert (guide.description != "(see script)").all(), "a column has no description"

    # ---- 3.13 save ---------------------------------------------------------
    out.to_csv(os.path.join(OUT_DIR, "twitter_clean.csv"), index=False)
    acct.to_csv(os.path.join(OUT_DIR, "twitter_clean_accounts.csv"), index=False)
    conflicts.to_csv(os.path.join(OUT_DIR, "label_conflicts.csv"), index=False)
    guide.to_csv(os.path.join(OUT_DIR, "column_guide.csv"), index=False)

    log(f"\n[SAVE]")
    log(f"          twitter_clean.csv           {out.shape[0]:,} rows x {out.shape[1]} cols  (one row per judged tweet)")
    log(f"          twitter_clean_accounts.csv  {acct.shape[0]:,} rows x {acct.shape[1]} cols  (one row per account)")
    log(f"          label_conflicts.csv         {conflicts.shape[0]:,} rows  ({conflicts.account_id.nunique()} accounts)")
    log(f"          column_guide.csv            {guide.shape[0]} columns documented")

    with open(os.path.join(OUT_DIR, "data_quality_report.txt"), "w") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\n[SAVE]     data_quality_report.txt")


if __name__ == "__main__":
    main()
