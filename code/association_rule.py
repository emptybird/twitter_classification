
"""
CSCI946 Assignment 2 - Association Rules
Author: Man Long Law 

Location : twitter_classification-main/code/association_rule.py

Input : ../output/twitter_clean_accounts.csv
        or
        Run AFTER prepare_data.py has created output/twitter_clean_accounts.csv:

            cd twitter_classification-main/code
            pip install mlxtend          (first time only)
            python association_rule.py

Output: ../output/association_rules_label.csv
        ../output/figures/fig*_ar_*.png 
"""

# Import packages 
import os
import numpy as np
 
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
 
import matplotlib.pyplot as plt
 
import warnings
warnings.filterwarnings("ignore")
 
HERE = os.path.dirname(os.path.abspath(__file__))       # .../twitter_classification-main/code
OUT = os.path.join(HERE, "..", "output")                # .../twitter_classification-main/output
FIG = os.path.join(OUT, "figures")                      # .../twitter_classification-main/output/figures
os.makedirs(FIG, exist_ok=True)

# Save everything that is printed into output/associate_rule_results.txt as well
import sys

class Tee:
    """write to the screen AND to a file at the same time"""
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.screen = sys.stdout
    def write(self, text):
        self.screen.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.screen.flush()
        self.file.flush()

sys.stdout = Tee(os.path.join(OUT, "associate_rule_results.txt"))
 
# load data
 
df = pd.read_csv(os.path.join(OUT, "twitter_clean_accounts.csv"), low_memory=False)
print(df.head())
 
# learn data
# Remove the accounts with no label (is_human is NaN)
# A rule cannot say "-> brand" or "-> human" about an account nobody labelled
 
df = df[df['label_known'] == 1]
print(df.shape)
print("human:", (df['is_human'] == 1).sum(), " brand:", (df['is_human'] == 0).sum())
 
# build the basket: one row per account
# The basket is built column by column
# Every item must be 0/1 for Apriori,and each data type is turned into 0/1 differently (Task 2 "data types"):
 
basket = pd.DataFrame(index=df.index)
 
# flags - already 0/1
basket['has_description'] = df['has_description']
basket['no_description'] = 1 - df['has_description']
basket['has_retweets'] = df['has_retweets']
basket['sidebar_is_zero'] = df['sidebar_color_zero_suspect']   # raw value was "0"
 
# counts - keep the raw count for now, encode_units turns them into 0/1 below
basket['tweet_has_url'] = df['tweet_n_urls']
basket['tweet_has_mention'] = df['tweet_n_mentions']
basket['tweet_has_hashtag'] = df['tweet_n_hashtags']
basket['tweet_has_emoji'] = df['tweet_n_mojibake']        # corrupted emoji runs
basket['description_has_url'] = df['description_n_urls']
basket['description_has_hashtag'] = df['description_n_hashtags']
 
# continuous - cut at the 25th / 75th percentile into "low" / "high" items
continuous = {'tweets_per_day': 'tweets_per_day',
              'favs_per_day': 'favs_per_day',
              'log_tweet_count': 'tweet_count',
              'log_account_age_days': 'account_age',
              'description_len': 'description_len',
              'tweet_len': 'tweet_len',
              'link_color_brightness': 'link_brightness',
              'sidebar_color_brightness': 'sidebar_brightness'}
for col, name in continuous.items():
    values = df[col].fillna(df[col].median())
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    basket[name + '_high'] = (values > q3).astype(int)
    basket[name + '_low'] = (values <= q1).astype(int)
    print("%-20s low <= %8.2f   high > %8.2f" % (name, q1, q3))
 
# the LABEL as two items, so rules can be read as "features -> label"
basket['human'] = (df['is_human'] == 1).astype(int)
basket['brand'] = (df['is_human'] == 0).astype(int)
 
# take a look at the processed data
print(basket.head())
 
# transform data - define function
 
def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
 
# transform data - execute function
basket_sets = basket.map(encode_units).astype(bool)
 
print(basket_sets.shape)
print(basket_sets.mean().sort_values(ascending=False))     # support of each single item
 
# frequent itemsets
# min_support=0.05 = an itemset must describe at least ~880 accounts
# max_len=4 keeps the rules short enough to read
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True, max_len=4)
 
print(frequent_itemsets.shape)
print(frequent_itemsets.sort_values('support', ascending=False).head(10))
 
# generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
 
print(rules.shape)
print(rules.head())
 
# Rules settings
# The right side must be exactly {human} or {brand}
rules = rules[rules['consequents'].isin([frozenset(['human']), frozenset(['brand'])])]
 
# The left side cannot contain labels.
rules = rules[~rules['antecedents'].apply(lambda s: 'human' in s or 'brand' in s)]
 
# add a new column with values being human or brand
rules['label'] = rules['consequents'].apply(lambda s: list(s)[0])
 
# convert antecedents to string
rules['antecedents'] = rules['antecedents'].apply(lambda s: ', '.join(sorted(s)))
 
print(rules.shape[0], "rules predict the label")
rules.sort_values('lift', ascending=False).to_csv(
    os.path.join(OUT, "association_rules_label.csv"), index=False)
 
# rules based on lift>=6 and confident>=0.8
lab_rules = rules[ (rules['lift'] >= 6) &
                   (rules['confidence'] >= 0.8) ]
print(len(lab_rules), "rules with lift >= 6 and confidence >= 0.8  (lab threshold -> too strict for this data)")
 
# base rate of each label (a rule is only useful if its confidence is above this)
print("P(human) =", basket_sets['human'].mean())
print("P(brand) =", basket_sets['brand'].mean())
 
# rules based on lift>=1.5 and confidence>=0.8, for brand
brand_rules = rules[ (rules['label'] == 'brand') &
                     (rules['lift'] >= 1.5) &
                     (rules['confidence'] >= 0.8) ]
brand_rules = brand_rules.sort_values('lift', ascending=False)
print(brand_rules[['antecedents', 'label', 'support', 'confidence', 'lift']].head(12))
 
# rules based on lift>=1.2 and confidence>=0.9, for human
# (human base rate is already 0.694, so a higher confidence is needed to be interesting)
human_rules = rules[ (rules['label'] == 'human') &
                     (rules['lift'] >= 1.2) &
                     (rules['confidence'] >= 0.9) ]
human_rules = human_rules.sort_values('lift', ascending=False)
print(human_rules[['antecedents', 'label', 'support', 'confidence', 'lift']].head(12))
 
# put the two sets of strong rules together for the next step
strong = pd.concat([brand_rules, human_rules])
 
# examine the strongest brand rule closely (Lab 6 did this with the alarm clocks)
top = brand_rules.iloc[0]                      # first row = highest lift (already sorted)
print(top['antecedents'], '->', top['label'])
 
# split the antecedent string back into a list of item names
items = top['antecedents'].split(', ')
print(items)
 
# for each account, check if it has ALL the items in the rule
match = basket_sets[items].all(axis=1)
print("accounts with all these items:", match.sum())
 
# how many of those accounts are brand, and how many are human?
print("of which brand:", (match & basket_sets['brand']).sum())
print("of which human:", (match & basket_sets['human']).sum())
 
# which accounts break a strong rule?
broken_ids = []
for i in range(len(strong)):
    r = strong.iloc[i]
    items = r['antecedents'].split(', ')
    other = 'human' if r['label'] == 'brand' else 'brand'
    match = basket_sets[items].all(axis=1) & basket_sets[other]
    broken_ids.extend(df.loc[match, 'account_id'])
 
# count how many rules each account breaks
cand = pd.Series(broken_ids, name='account_id').value_counts().reset_index(name='n_rules_broken')
cand = cand.rename(columns={'index': 'account_id'})
cand = cand.merge(df[['account_id', 'name', 'gender', 'gender_confidence', 'is_human']], on='account_id')
 
print("accounts breaking at least one strong rule:", len(cand))
print("labelled human but match a brand rule:", (cand['is_human'] == 1).sum())
print("labelled brand but match a human rule:", (cand['is_human'] == 0).sum())
 
others = df[~df['account_id'].isin(cand['account_id'])]
print("mean gender_confidence, rule-breakers:", cand['gender_confidence'].mean())
print("mean gender_confidence, others:       ", others['gender_confidence'].mean())
 
print(cand.head(10))
cand.to_csv(os.path.join(OUT, "candidate_mislabels_rules.csv"), index=False)
 
# Task3
# Figure 1 - support vs confidence, the standard rule plot
 
plt.figure(figsize=(7, 5))
for label, colour in [('human', 'tab:blue'), ('brand', 'tab:orange')]:
    d = rules[rules['label'] == label]
 
    plt.scatter(d['support'], d['confidence'], s=10 + 40 * (d['lift'] - 1), alpha=0.45,
                color=colour, label='-> ' + label)
 
plt.axhline(basket_sets['human'].mean(), color='tab:blue', ls=':', lw=0.9)
plt.axhline(basket_sets['brand'].mean(), color='tab:orange', ls=':', lw=0.9)
plt.xlabel('support'); plt.ylabel('confidence'); plt.legend(loc='lower right')
plt.title('Rules predicting the label (dotted = base rate, size = lift)')
plt.savefig(os.path.join(FIG, "fig1_ar_support_confidence.png"), dpi=200, bbox_inches="tight")
plt.show()
 
# Figure 2 - strongest rules by lift, as a bar chart
top = pd.concat([brand_rules.sort_values('lift', ascending=False).head(8),
                 human_rules.sort_values('lift', ascending=False).head(8)])
 
plt.figure(figsize=(9, 0.42 * len(top) + 1.2))
labels = ['{%s} -> %s' % (a, l) for a, l in zip(top['antecedents'], top['label'])]
plt.barh(labels, top['confidence'], color=['tab:orange' if l == 'brand' else 'tab:blue' for l in top['label']])
 
for i, (c, lf) in enumerate(zip(top['confidence'], top['lift'])):
    plt.text(c + 0.005, i, 'conf %.2f  lift %.2f' % (c, lf), va='center', fontsize=8)
 
plt.gca().invert_yaxis()
plt.xlim(0, 1.25); plt.xlabel('confidence')
plt.title('Strongest rules (orange -> brand, blue -> human)')
plt.savefig(os.path.join(FIG, "fig2_ar_strongest_rules_by_lift.png"), dpi=200, bbox_inches="tight")
plt.show()
 
# Figure 3 - single-item support, human vs brand
items_only = basket_sets.drop(columns=['human', 'brand'])
supp = pd.DataFrame({'human': items_only[basket_sets['human']].mean(),
                     'brand': items_only[basket_sets['brand']].mean()})
supp = supp.assign(diff=supp['brand'] - supp['human']).sort_values('diff')
y = np.arange(len(supp))
 
plt.figure(figsize=(7.5, 6.5))
plt.barh(y - 0.2, supp['human'], height=0.4, label='human accounts')
plt.barh(y + 0.2, supp['brand'], height=0.4, label='brand accounts')
plt.yticks(y, supp.index, fontsize=8); plt.xlabel('fraction of accounts having the item'); plt.legend()
plt.title('Item support by label')
plt.savefig(os.path.join(FIG, "fig3_ar_single-item_support.png"), dpi=200, bbox_inches="tight")
plt.show()

# the Tee object
results = sys.stdout            

# restore the original screen output
sys.stdout = results.screen     
results.file.close()









