from pathlib import Path
import hashlib

import pandas as pd
import sklearn
import aif360

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "compas_farayola_numeric.csv"
)

data = pd.read_csv(DATA_PATH)

# Confirm that this is the reconstructed dataset we expect.
assert data.shape == (5278, 12), f"Unexpected dataset shape: {data.shape}"

assert data["two_year_recid"].value_counts().to_dict() == {
    0: 2795,
    1: 2483,
}, "Unexpected target counts"

assert data["race"].value_counts().to_dict() == {
    0: 3175,
    1: 2103,
}, "Unexpected race counts"


# ---------------------------------------------------------------------
# Predictors and target
# ---------------------------------------------------------------------

X = data.drop(columns=["two_year_recid"])

# Use a 1D Series here because that is scikit-learn's expected target
# representation. This does not change the observations or labels.
y = data["two_year_recid"]


# ---------------------------------------------------------------------
# Train/test split
#
# Matches the split parameters visible in the IF-EA COMPAS notebook:
# 70/30, random_state=101, shuffle=True, stratified on target y.
# ---------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101,
    shuffle=True,
    stratify=y,
)


# ---------------------------------------------------------------------
# Candidate NONE baseline
#
# IMPORTANT:
# The paper does not clearly identify the classifier used for the NONE
# condition. Logistic Regression is being tested as a candidate because
# it is the base classifier explicitly used with EGR in the paper/code.
#
# No fairness intervention is applied here.
# ---------------------------------------------------------------------

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# ---------------------------------------------------------------------
# Construct AIF360 datasets using the same semantic conventions used
# by the IF-EA COMPAS notebook.
# ---------------------------------------------------------------------

test_df = pd.concat(
    [
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    ],
    axis=1,
)

pred_df = test_df.copy()
pred_df["two_year_recid"] = y_pred.astype(int)

favorable_label = 1
unfavorable_label = 0

protected_attribute_names = ["race"]

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

dataset_test_bld = BinaryLabelDataset(
    df=test_df,
    label_names=["two_year_recid"],
    favorable_label=favorable_label,
    unfavorable_label=unfavorable_label,
    protected_attribute_names=protected_attribute_names,
    privileged_protected_attributes=privileged_group,
)

predictions_bld = BinaryLabelDataset(
    df=pred_df,
    label_names=["two_year_recid"],
    favorable_label=favorable_label,
    unfavorable_label=unfavorable_label,
    protected_attribute_names=protected_attribute_names,
    privileged_protected_attributes=privileged_group,
)


# ---------------------------------------------------------------------
# Metrics
#
# Unlike the erroneous positional call we found in Run 001, use named
# group arguments here so there is no ambiguity about orientation.
# ---------------------------------------------------------------------

metric = ClassificationMetric(
    dataset_test_bld,
    predictions_bld,
    unprivileged_groups=unprivileged_group,
    privileged_groups=privileged_group,
)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])


# ---------------------------------------------------------------------
# Reproducibility information
# ---------------------------------------------------------------------

sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

print("=" * 70)
print("RUN 002 — CANDIDATE NONE BASELINE USING LOGISTIC REGRESSION")
print("=" * 70)

print()
print("IMPORTANT:")
print(
    "Logistic Regression is a candidate baseline classifier; "
    "the paper does not clearly specify the classifier behind NONE."
)

print()
print("INPUT")
print("-----")
print("File:", DATA_PATH)
print("SHA256:", sha256)
print("Shape:", data.shape)

print()
print("TARGET COUNTS")
print("-------------")
print(data["two_year_recid"].value_counts().sort_index())

print()
print("RACE COUNTS")
print("-----------")
print(data["race"].value_counts().sort_index())

print()
print("TRAIN / TEST")
print("------------")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Split random_state:", 101)
print("Stratified on:", "two_year_recid")

print()
print("CLASSIFIER")
print("----------")
print("Classifier:", classifier)
print("Solver:", classifier.solver)
print("Configured max_iter:", classifier.max_iter)
print("Iterations actually used:", classifier.n_iter_)

print()
print("CONFUSION MATRIX")
print("----------------")
print("Rows = actual [0, 1]")
print("Cols = predicted [0, 1]")
print(cm)

print()
print("RESULTS")
print("-------")
print("Accuracy: {:.12f}".format(accuracy))
print("Disparate Impact: {:.12f}".format(metric.disparate_impact()))
print("Mean Difference: {:.12f}".format(metric.mean_difference()))
print(
    "Statistical Parity Difference: {:.12f}".format(
        metric.statistical_parity_difference()
    )
)
print(
    "Equal Opportunity Difference: {:.12f}".format(
        metric.equal_opportunity_difference()
    )
)
print(
    "Predictive Equality Difference: {:.12f}".format(
        metric.false_positive_rate_difference()
    )
)

print()
print("GROUP-DIFFERENCE DETAILS")
print("------------------------")
print(
    "True Positive Rate Difference "
    "(Unprivileged - Privileged): {:.12f}".format(
        metric.true_positive_rate_difference()
    )
)
print(
    "False Positive Rate Difference "
    "(Unprivileged - Privileged): {:.12f}".format(
        metric.false_positive_rate_difference()
    )
)

print()
print("SOFTWARE")
print("--------")
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("aif360:", aif360.__version__)
