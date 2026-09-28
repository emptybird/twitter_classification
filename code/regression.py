
"""
CSCI946 Assignment 2 - Regression
Author: Man Long Law 

Location : twitter_classification-main/code/regression.py

Input : ../output/twitter_clean_accounts.csv
        or
        Run AFTER prepare_data.py has created output/twitter_clean_accounts.csv:

            cd twitter_classification-main/code
            python regression.py

Output: ../output/candidate_mislabels_logreg.csv
        ../output/figures/fig*_reg_*.png
"""

# Import packages
import os

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pylab as plt


from sklearn.preprocessing import StandardScaler
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# paths are relative to THIS file, so the script works from any computer as
# long as the folder layout is  code/<this file>  and  output/<csv files>
HERE = os.path.dirname(os.path.abspath(__file__))       # .../twitter_classification-main/code
OUT = os.path.join(HERE, "..", "output")                # .../twitter_classification-main/output
FIG = os.path.join(OUT, "figures")                      # .../twitter_classification-main/output/figures
os.makedirs(FIG, exist_ok=True)

# Save everything that is printed into output/regression_results.txt as well
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

sys.stdout = Tee(os.path.join(OUT, "regression_results.txt"))

# 1. load data
# Load the data and show the info and contents
acc = pd.read_csv(os.path.join(OUT, "twitter_clean_accounts.csv"), low_memory=False)
print(acc.head())

# 2. summarize the dataset
print(acc.shape)
print(acc.describe())

# Check how many classes we do have from the "is_human" column
print(set(acc['is_human'].dropna()))

# Check number of samples for each class and comment whether dataset is balanced
# 1 = human (female/male), 0 = brand, NaN = unknown
print("No. of human samples:  ", acc[acc['is_human'] == 1].shape[0])
print("No. of brand samples:  ", acc[acc['is_human'] == 0].shape[0])
print("No. of unknown samples:", acc['is_human'].isna().sum())

# about 69% human vs 31% brand, so the dataset is NOT balanced (see step8)

# 3. clean your data
# the unknown-label rows are NOT dropped, they are set aside and scored later
# because "unknown" is where mislabelling is likely to hide.
print(acc.isna().sum()[acc.isna().sum() > 0])
unknown = acc[acc['label_known'] == 0].copy()
data = acc[acc['label_known'] == 1].copy()
print(data.shape)

# Features. Chosen by DATA TYPE (Task 2): behaviour counts, colour, text
# Here gender, gender_confidence and label_conflict_* must be left out
# because they ARE the label (see column_guide.csv) - using them would leak
features = [
    'log_tweet_count', 'log_account_age_days', 'tweets_per_day', 'favs_per_day', 'has_retweets', # behaviour (numeric)
    'link_color_brightness', 'sidebar_color_brightness',
    'link_color_r', 'link_color_g', 'link_color_b', 'sidebar_color_zero_suspect', ## colour (numeric 0-255 + flag)
    'has_description', 'description_len', 'description_word_count', # text
    'description_n_urls', 'description_n_mentions', 'description_n_hashtags',
    'tweet_len', 'tweet_word_count', 'tweet_n_urls',
    'tweet_n_mentions', 'tweet_n_hashtags', 'tweet_n_mojibake']

# Deal with the NaN values in the data: only 2 accounts have an unrepairable link colour
# Fill with the median rather than drop, so no label is lost
for col in features:
    median_value = data[col].median()
    data[col] = data[col].fillna(median_value)
    unknown[col] = unknown[col].fillna(median_value)

print("")
print("NaN left in features:", data[features].isna().sum().sum())

# 4. normalize data
print(data[['tweets_per_day', 'favs_per_day']].describe())

data['tweets_per_day'] = np.log1p(data['tweets_per_day'])
data['favs_per_day'] = np.log1p(data['favs_per_day'])
unknown['tweets_per_day'] = np.log1p(unknown['tweets_per_day'])
unknown['favs_per_day'] = np.log1p(unknown['favs_per_day'])

print(data[['tweets_per_day', 'favs_per_day']].describe())

# 5. Visualize the linear relationship (Task 3 - Figure 1)
# Plot log_tweet_count vs log_fav_number
sns.lmplot(x="log_tweet_count", y="log_fav_number", data=data.sample(3000, random_state=142),
           hue="gender", height=5.2, aspect=1.4, scatter_kws={'s': 6, 'alpha': 0.4})

plt.savefig(os.path.join(FIG, "fig1_reg_lmplot_tweets_vs_favs.png"), dpi=200, bbox_inches="tight")
plt.show()

# 6. Split your data into training (80%) and testing data (20%),random_state=142  (same as Lab 5)
train, test = train_test_split(data, test_size=0.2, random_state=142)
print(train.shape)
print(test.shape)

# 7. simple Linear regression
# target: log_fav_number   (how many favourites an account gives)
# favs_per_day = fav_number / age, i.e. the target in disguise, so exclude it
lin_features = []
for f in features:
    if f != 'favs_per_day':
        lin_features.append(f)

x = train[lin_features]
y = train[['log_fav_number']]
est = LinearRegression(fit_intercept=True)
est.fit(x, y)

print("Coefficients:", est.coef_)
print("Intercept:", est.intercept_)

model = LinearRegression()
model.fit(x, y)
x_test = test[lin_features]
y_test = test[['log_fav_number']]
y_hat = model.predict(x_test)

print("MSE:", metrics.mean_squared_error(y_test, y_hat))
print("R^2:", metrics.r2_score(y_test, y_hat))
print("var:", y_test.var())

# residual = actual - predicted
# Do brands get fewer favourites than predicted?
test['resid'] = y_test['log_fav_number'] - y_hat[:, 0]
print(test.groupby('gender')['resid'].mean())

lin_y_test = y_test
lin_y_hat = y_hat

# 8. load and summarise data, build and test logistic regression model
# target: is_human
# Getting input data and targets for building prediction model
X_train = train[features]
y_train = train['is_human'].astype(int)
X_test = test[features]
y_test = test['is_human'].astype(int)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train) # learn mean and std from training data only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Build your Logistic Regression model
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# Do predictions on test set
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)

# 9. Evaluate the performance of the model
print("Accuracy score on training set: ", accuracy_score(y_train, y_hat_train))
print("Accuracy score on testing set: ", accuracy_score(y_test, y_hat_test))

# Checking confusion matrix
print("Confusion matrix on train set: ")
print(confusion_matrix(y_train, y_hat_train))
print("Confusion matrix on test set: ")
print(confusion_matrix(y_test, y_hat_test))

# Which features push towards human (+) or brand (-)?
coef = pd.Series(model.coef_[0], index=features)
coef = coef.sort_values()
print(coef)

# P(human) for each test account, and the area under the ROC curve
p_test = model.predict_proba(X_test)[:, 1]         # P(human) for each test account
auc = roc_auc_score(y_test, p_test)
print("ROC AUC on test set:", auc)

# 10. feature selection with RFE (3, 5 and 8 features)
for n in [3, 5, 8]:
    lr_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    rfe = RFE(estimator=lr_model, n_features_to_select=n, step=1)
    rfe.fit(X_train, y_train)
    y_test_hat = rfe.predict(X_test)
    print("\n%d features - accuracy score on test set: %.4f" % (n, accuracy_score(y_test, y_test_hat)))

# summarize all features
    for i in range(X_train.shape[1]):
        if rfe.support_[i]:
            print('Column: %s, Selected %s, Rank: %.3f' % (features[i], rfe.support_[i], rfe.ranking_[i]))

# 11. Conclusion
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if (y_actual[i] == y_hat[i]) and (y_hat[i] == 1):
            TP += 1
        if (y_hat[i] == 1) and (y_actual[i] != y_hat[i]):
            FP += 1
        if (y_actual[i] == y_hat[i]) and (y_hat[i] == 0):
            TN += 1
        if (y_hat[i] == 0) and (y_actual[i] != y_hat[i]):
            FN += 1
    return TP, FP, TN, FN

TP, FP, TN, FN = perf_measure(np.array(y_test), np.array(y_hat_test))
print('%d True Positive, %d False Positive, %d True Negative, and %d False Negative' % (TP, FP, TN, FN))

# positive = human. FP = a brand the model calls human; FN = a human the model calls brand

# 12. which profiles look mislabelled
# If the model says P(human) >= 0.85 for an account recorded as brand, or P(human) <= 0.15 for one recorded as human,
# the model and the annotators strongly disagree -> candidate mislabel.
# To score ALL labelled accounts (not just the 20% test set) without letting
# any account be scored by a model that saw its own label, use 5-fold
# cross-validation: cross_val_predict fits 5 models, each on 4/5 of the data,
# and predicts the remaining 1/5. Every account is therefore predicted by a
# model that never saw it. (Same idea as the train/test split, repeated 5x.)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

X_all = scaler.transform(data[features])
y_all = data['is_human'].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
cv_model = LogisticRegression(max_iter=2000, class_weight='balanced')

p_all = cross_val_predict(cv_model, X_all, y_all, cv=cv, method='predict_proba')
data['p_human'] = p_all[:, 1]
y_all_hat = (data['p_human'] >= 0.5).astype(int)

print("\ncross-validated accuracy on all labelled accounts:", accuracy_score(y_all, y_all_hat))

# disagreement = how far the model's probability is from the recorded label
# labelled human (1): disagreement = 1 - p_human
# labelled brand (0): disagreement = p_human
data['disagreement'] = data['p_human']                                  # start with brand case
data.loc[data['is_human'] == 1, 'disagreement'] = 1 - data['p_human']   # overwrite human case

cand = data[data['disagreement'] >= 0.85]
cand = cand.sort_values('disagreement', ascending=False)

print("candidate mislabels:", len(cand), "of", len(data))
print("  labelled human, model says brand:", (cand['is_human'] == 1).sum())
print("  labelled brand, model says human:", (cand['is_human'] == 0).sum())

# gender_confidence was NOT a feature, so if candidates have lower confidence,
# the annotators were also unsure about them -> independent evidence
others = data[data['disagreement'] < 0.85]

print("  mean gender_confidence, candidates:", cand['gender_confidence'].mean())
print("  mean gender_confidence, others:    ", others['gender_confidence'].mean())
print(cand[['name', 'gender', 'gender_confidence', 'p_human', 'description_raw']].head(10))

cand[['account_id', 'name', 'gender', 'gender_confidence', 'is_human', 'p_human',
      'disagreement', 'description_raw', 'tweet_raw']].to_csv(
    os.path.join(OUT, "candidate_mislabels_logreg.csv"), index=False)

# and what does the model think the 'unknown' accounts are?
X_unknown = scaler.transform(unknown[features])
unknown['p_human'] = model.predict_proba(X_unknown)[:, 1]

print("unknown accounts: model calls", (unknown['p_human'] >= 0.5).sum(), "human and",
      (unknown['p_human'] < 0.5).sum(), "brand")

# Task 3 - Visualisation for evaluation
# Figure 1 - the linear relationship(step5): log_tweet_count vs log_fav_number
# This figure is shown before

# Figure 2 - linear regression(step7): predicted vs actual on the test set
# plot predicted vs actual
plt.figure(figsize=(6, 5))
plt.plot(lin_y_hat, lin_y_test, 'o', alpha=0.3, markersize=3)
plt.plot([0, 13], [0, 13], 'r', alpha=0.7)          # the "perfect prediction" line
plt.xlabel('predicted log(1+fav_number)')
plt.ylabel('actual log(1+fav_number)')
plt.title('Linear regression on test set')
plt.savefig(os.path.join(FIG, "fig2_reg_linear_pred_vs_actual.png"), dpi=200, bbox_inches="tight")
plt.show()

# Figure 3 - logistic regression(step9): confusion matrix heatmap + ROC curve
fpr, tpr, thresholds = roc_curve(y_test, p_test)
plt.figure(figsize=(10, 4.2))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_hat_test), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['pred brand', 'pred human'], yticklabels=['actual brand', 'actual human'])
plt.title('Confusion matrix (test)')
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label='logistic regression')
plt.plot([0, 1], [0, 1], 'k--', lw=0.8, label='random guess')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()
plt.title('ROC curve (test), AUC = %.2f' % auc)
plt.savefig(os.path.join(FIG, "fig3_reg_confusion_roc.png"), dpi=200, bbox_inches="tight")
plt.show()

# Figure 4 - P(human) by recorded label. Shaded = candidate-mislabel zone.
# histogram of P(human) by recorded label (Task 3). Shaded = candidate zone.
plt.figure(figsize=(7, 4.2))
plt.hist(data[data['is_human'] == 1]['p_human'], bins=40, alpha=0.6, label='labelled human')
plt.hist(data[data['is_human'] == 0]['p_human'], bins=40, alpha=0.6, label='labelled brand')
plt.hist(unknown['p_human'], bins=40, histtype='step', color='k', label='labelled unknown')
plt.axvspan(0, 0.15, color='grey', alpha=0.15)
plt.axvspan(0.85, 1, color='grey', alpha=0.15)
plt.xlabel('cross-validated P(human) from logistic regression')
plt.ylabel('number of accounts')
plt.legend()
plt.title('Shaded: model disagrees with the label -> candidate mislabel')
plt.savefig(os.path.join(FIG, "fig4_reg_candidate_mislabels.png"), dpi=200, bbox_inches="tight")
plt.show()

print("\nfigures written to", FIG)

# the Tee object
results = sys.stdout            

# restore the original screen output
sys.stdout = results.screen     
results.file.close()
