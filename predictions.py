import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, KFold

try:
    with open('pickle_df/prepared_df.pkl', 'rb') as pickle_file:
        loans = pd.read_pickle(pickle_file)
except Exception as e:
    print("Please make sure to run preparing_data.py first")

penalty = {
    0: 10,
    1: 1
}
lr = LogisticRegression(class_weight=penalty)
cols = loans.columns
train_cols = cols.drop('loan_status')
features = loans[train_cols]
target = loans['loan_status']
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)
