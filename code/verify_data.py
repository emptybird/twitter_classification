"""
CSCI946 Assignment 2 - Independent verification of the prepared data
Author: Tianyu He (9495204)

Run AFTER prepare_data.py:   python verify_data.py

This script deliberately shares no code with prepare_data.py. It recomputes
every claim from the raw CSV and compares against the written outputs, so a
bug in the pipeline cannot hide behind the pipeline's own helper functions.
Exit code is 0 only when every check passes.
"""

import os
import re
import sys
import glob
import html
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output")


def find_raw():
    for c in [os.path.join(HERE, "..", "A2_2026_Released"),
              os.path.join(HERE, "A2_2026_Released"), HERE, os.path.join(HERE, "..")]:
        hit = glob.glob(os.path.join(c, "twitter_user_data.csv"))
        if hit:
            return hit[0]
    raise FileNotFoundError("twitter_user_data.csv not found")


results = []


def check(section, name, condition, detail=""):
    ok = bool(condition)
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def main():
    raw = pd.read_csv(find_raw(), encoding="latin-1", low_memory=False)
    row = pd.read_csv(os.path.join(OUT_DIR, "twitter_clean.csv"), low_memory=False)
    acc = pd.read_csv(os.path.join(OUT_DIR, "twitter_clean_accounts.csv"), low_memory=False)
    cnf = pd.read_csv(os.path.join(OUT_DIR, "label_conflicts.csv"), low_memory=False)
    gid = pd.read_csv(os.path.join(OUT_DIR, "column_guide.csv"))
    kept = raw[raw.profile_yn == "yes"]

    print("\nA. Row counts and keys")
    check("A", "row table = 20,050 - 97 unreadable profiles",
          len(row) == len(raw) - 97, f"{len(row):,}")
    check("A", "unit_id is unique", row.unit_id.duplicated().sum() == 0)
    check("A", "account table = distinct account_id in row table",
          len(acc) == row.account_id.nunique(), f"{len(acc):,}")
    check("A", "account_id is unique in the account table",
          acc.account_id.duplicated().sum() == 0)
    check("A", "no row was silently dropped besides the 97",
          set(row.unit_id) == set(kept._unit_id))

    print("\nB. Labels")
    check("B", "human count matches the raw file",
          int((row.is_human == 1).sum()) == int(kept.gender.isin(["male", "female"]).sum()),
          f"{int((row.is_human == 1).sum()):,}")
    check("B", "non-human count matches the raw file",
          int((row.is_human == 0).sum()) == int((kept.gender == "brand").sum()),
          f"{int((row.is_human == 0).sum()):,}")
    check("B", "unknown rows are kept, not deleted",
          int(row.is_human.isna().sum()) == int((kept.gender == "unknown").sum()),
          f"{int(row.is_human.isna().sum()):,}")
    k = kept.copy()
    k["h"] = k.gender.map({"female": 1, "male": 1, "brand": 0})
    k["aid"] = k.name.astype(str) + "|" + k.created.astype(str)
    exp_hb = int((k.dropna(subset=["h"]).groupby("aid").h.nunique() > 1).sum())
    exp_any = int((k.groupby("aid").gender.nunique() > 1).sum())
    check("B", "human/brand conflicting accounts",
          cnf[cnf.label_conflict_human_brand == 1].account_id.nunique() == exp_hb, f"{exp_hb}")
    check("B", "all conflicting accounts exported",
          cnf.account_id.nunique() == exp_any, f"{exp_any}")

    print("\nC. Colour repair")
    s = kept.link_color.astype(str).str.strip().str.upper()
    check("C", "zero_suspect count equals raw '0' count",
          int((row.link_color_zero_suspect == 1).sum()) == int((s == "0").sum()),
          f"{int((s == '0').sum())}")
    check("C", "sci_repaired count equals raw sci-notation count (exponent >= 2)",
          int((row.link_color_status == "sci_repaired").sum())
          == int(s.str.match(r"^[0-9]\.[0-9]{2}E\+(?:0[2-9]|[1-9][0-9])$").sum()))
    # A VALUE check, not a count: the repaired hex, read back as a number,
    # must equal the number the spreadsheet stored. The previous version of
    # this file only checked that the repair was valid hex, which let a
    # numerically wrong repair pass.
    for col in ["link_color", "sidebar_color"]:
        rs = kept[col].astype(str).str.strip().str.upper().values
        m = (row[f"{col}_status"] == "sci_repaired").values
        pairs = list(zip(row.loc[m, f"{col}_hex"], rs[m]))
        check("C", f"{col}: every sci_repaired hex equals the raw number",
              len(pairs) > 0 and all(np.isclose(float(h), float(r)) for h, r in pairs),
              f"{len(pairs)} values")
    rgb_cols = [c for c in row.columns if c.endswith(("_r", "_g", "_b"))]
    check("C", "every RGB value is within 0-255",
          bool(row[rgb_cols].stack().dropna().between(0, 255).all()))
    ok_row = row[row.link_color_status == "ok"].iloc[0]
    h = ok_row.link_color_hex
    check("C", f"hand-computed {h} -> R{int(h[0:2],16)} G{int(h[2:4],16)} B{int(h[4:6],16)}",
          (ok_row.link_color_r, ok_row.link_color_g, ok_row.link_color_b)
          == (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
    check("C", "usable flag agrees with a null hex",
          ((row.link_color_hex.isna()) == (row.link_color_usable == 0)).all())

    print("\nD. Text cleaning")
    for p in ["tweet", "description"]:
        c = row[f"{p}_clean"].fillna("")
        check("D", f"{p}_clean has no non-ASCII characters",
              not c.str.contains(r"[^\x00-\x7F]").any())
        check("D", f"{p}_clean has no surviving URL", not c.str.contains("http", case=False).any())
        # Idempotency, not a pattern match: text such as 'ig&phhhoto;' merely
        # looks like an entity and must be left alone. The real requirement is
        # that another unescape pass would change nothing.
        check("D", f"{p}_clean has no decodable HTML entity left",
              (c.apply(html.unescape) == c).all())
        check("D", f"{p}_len equals len({p}_clean)", (row[f"{p}_len"] == c.str.len()).all())
        # A lone '_' token may only survive where the author typed one
        # ('think of _____'); any other lone '_' is left-over mojibake.
        lone = r"(?:^|\s)_+(?=\s|$)"
        src = kept["text" if p == "tweet" else "description"].fillna("").astype(str)
        check("D", f"{p}_clean has no stray '_' token left by mojibake",
              not (c.str.contains(lone).values
                   & ~src.reset_index(drop=True).str.contains(lone).values).any())
    check("D", "tweet_raw is byte-identical to the source column",
          (row.tweet_raw.fillna("").values == kept.text.fillna("").astype(str).values).all())

    print("\nE. Engineered features")
    check("E", "log_tweet_count == log1p(tweet_count)",
          np.allclose(row.log_tweet_count, np.log1p(row.tweet_count)))
    check("E", "log_fav_number == log1p(fav_number)",
          np.allclose(row.log_fav_number, np.log1p(row.fav_number)))
    check("E", "tweets_per_day == tweet_count / max(age, 1)",
          np.allclose(row.tweets_per_day, row.tweet_count / row.account_age_days.clip(lower=1)))
    check("E", "account_age_days is never negative", bool((row.account_age_days >= 0).all()))
    check("E", "has_retweets == (retweet_count > 0)",
          (row.has_retweets == (row.retweet_count > 0).astype(int)).all())

    print("\nF. Account-table aggregation")
    mx = row.groupby("account_id")[["fav_number", "tweet_count", "retweet_count"]].max()
    a = acc.set_index("account_id").sort_index()
    check("F", "counts are the per-account maximum",
          np.allclose(a[["fav_number", "tweet_count", "retweet_count"]], mx.sort_index()))
    check("F", "log columns recomputed from the aggregated counts",
          np.allclose(acc.log_tweet_count, np.log1p(acc.tweet_count)))
    check("F", "account-constant columns survive aggregation",
          bool((acc.link_color_r.notna().sum() > 0)
               and np.isclose(acc.account_age_days.min(), row.account_age_days.min())))

    check("F", "account table: is_human agrees with gender on every row",
          bool(((acc.gender == "unknown") == acc.is_human.isna()).all()
               and (acc.loc[acc.gender == "brand", "is_human"] == 0).all()
               and (acc.loc[acc.gender.isin(["male", "female"]), "is_human"] == 1).all()))
    check("F", "account table: every row is a whole row of the row table",
          set(acc.unit_id) <= set(row.unit_id)
          and bool((acc.set_index("unit_id").gender
                    == row.set_index("unit_id").gender.reindex(acc.unit_id)).all()))

    print("\nG. Documentation")
    check("G", "column_guide covers every output column",
          set(gid.column) == set(row.columns), f"{len(gid)} columns")
    check("G", "every column has a real description",
          gid.description.notna().all() and (gid.description != "(see script)").all())

    passed, total = sum(results), len(results)
    print(f"\n{'=' * 60}\n{passed}/{total} checks passed")
    if passed != total:
        print("VERIFICATION FAILED")
        sys.exit(1)
    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
