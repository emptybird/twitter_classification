# CSCI946 Assignment 2 — Data Preparation

Cleaning pipeline for `twitter_user_data.csv` (CrowdFlower gender-classifier
dataset, 20,050 rows). Produces the modelling tables used by the rest of the
group for clustering, classification, regression, association rules and text
processing.

## Quick start

The source CSV is **not** in this repository — download `twitter_user_data.csv`
from Moodle and put it in a folder named `A2_2026_Released/` beside `code/`.
Then:

```bash
python code/prepare_data.py    # ~20 s, writes output/
python code/verify_data.py     # must print 40/40 checks passed
python code/make_figures.py    # regenerates the three report figures
```

If you only need the data, skip all of the above and read
`output/twitter_clean_accounts.csv` directly — it is committed.

`verify_data.py` shares no code with the pipeline: it recomputes every claim
from the raw CSV. If it does not print `VERIFICATION PASSED`, do not use the
data.

## Outputs

| File | Rows | What it is |
|---|---|---|
| `output/twitter_clean.csv` | 19,953 | One row per **judged tweet**. Committed to the repo. |
| `output/twitter_clean_accounts.csv` | 18,715 | One row per **account**. Committed to the repo. |
| `output/label_conflicts.csv` | 493 | Accounts whose rows carry contradictory labels. |
| `output/column_guide.csv` | 55 | Every output column, its type, missing count and meaning. |
| `output/data_quality_report.txt` | — | Full audit trail of the last run. |
| `output/data_preparation_report_section.md` | — | The Data Preparation section for the group report. |
| `output/figures/` | 3 | The figures referenced by that section. |

## READ THIS BEFORE SPLITTING THE DATA

`twitter_clean.csv` is **not** one row per account. 791 accounts appear on
more than one row (2,029 rows in total; one account appears 30 times), because
each row is an (account, sampled tweet) pair judged independently.

A random `train_test_split` therefore puts the same account on both sides and
every score comes out optimistic. Do one of:

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["account_id"]))
```

or use `twitter_clean_accounts.csv`, which is already deduplicated.

## Which columns to use

- **Target:** `is_human` — 1 = female/male, 0 = brand, NaN = unknown.
- **Never use as features:** `name`, `gender`, `gender_confidence`,
  `label_known`, `label_conflict_*`. They are the label or derived from it.
- **Distance-based models** (k-means, KNN): use the `log_` columns.
- **Tree models:** raw or `log_` columns both work — a monotone transform does
  not change an axis-parallel split.
- **Text models:** use `tweet_clean` / `description_clean`. The `*_raw`
  columns are the untouched source text.
- **Colour:** check `*_status`. 4,006 sidebar and 417 link values were the
  literal string `"0"`; they are stored as `000000` but flagged
  `zero_suspect`, because it cannot be determined whether they are real black.
  Run colour-based models both with and without them.

## Known limitations (state these in the report)

1. **Ambiguous colour repairs.** 8 of the 40 scientific-notation repairs have a
   mantissa ending in 0, so a reading with leading zeros is numerically
   identical. The no-leading-zero reading was chosen.
2. **`"0"` colours.** Cannot be shown to be real black; kept and flagged.
3. **Non-ASCII removal.** `*_clean` drops every non-ASCII character, including
   legitimate accented text and symbols such as `£`, not only corrupted emoji.
   Acceptable for English bag-of-words and TF-IDF.
4. **Account-level label.** For multi-row accounts the account table keeps the
   row with the highest `gender_confidence`. For the 157 conflicting accounts
   this is a judgement call, so every original row is kept in
   `label_conflicts.csv`.

## Repository is private

The assignment states that plagiarism of any part leads to zero marks for the
whole group. Keep this repository private and add members as collaborators.
