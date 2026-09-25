"""
Statistical analysis behind the notebook. Pure computation -- no plotting here.

evaluate.py prints point estimates. This module asks the follow-up questions a
reviewer should ask of any point estimate:

  * How sure are we? The test set holds 31 fraudulent invoices. A recall of 90%
    on 31 cases could plausibly be anywhere from the high 70s to 100%, so every
    headline number gets a bootstrap confidence interval.
  * Is the forest really better than the tree, or is that noise? A PAIRED bootstrap
    resamples the same invoices for both models, so the difference is measured on
    identical data.
  * Which features matter on data the model has never seen? Permutation importance
    on the held-out test set, the same measurement for every model, instead of each
    model's built-in importance (which is measured on training data and defined
    differently per model).
  * What does the agent do to an analyst's workload? Counts per routing decision.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from agent import ESCALATE_ABOVE, RELEASE_BELOW
from features import FEATURE_NAMES

SEED = 20261113
N_BOOTSTRAP = 2000
N_PERMUTATION_REPEATS = 30

# Display names, fixed once so every table and figure uses the same words.
TREE = "Decision tree (shipped)"
FOREST = "Random forest"
LOGISTIC = "Logistic regression"
NAIVE = "Decision tree + region"
RULE = "Keyword rule"


def keyword_rule(df):
    """The rule an AP team writes before anyone suggests machine learning."""
    return (
        (df["max_similarity_to_prior"] > 0.95) & (df["amount_ratio_to_closest_match"] > 0.99)
    ).astype(int).to_numpy()


def load():
    """The saved models, the full ledger, and the sealed test slice."""
    bundle = pickle.load(open("models/models.pkl", "rb"))
    test = pd.read_csv("data/test.csv").fillna({"po_number": "", "fraud_type": ""})
    train = pd.read_csv("data/train.csv").fillna({"po_number": "", "fraud_type": ""})
    return bundle, train, test


def with_region(df, region_cols):
    out = df[FEATURE_NAMES].copy()
    for r in region_cols:
        out[f"region={r}"] = (df["region"] == r).astype(float)
    return out


def scored_models(bundle, test):
    """Fraud probability from each model on the test set, keyed by display name."""
    X = test[FEATURE_NAMES]
    return {
        TREE: bundle["responsible"].predict_proba(X)[:, 1],
        FOREST: bundle["forest"].predict_proba(X)[:, 1],
        LOGISTIC: bundle["logistic"].predict_proba(X)[:, 1],
        NAIVE: bundle["naive"].predict_proba(with_region(test, bundle["region_cols"]))[:, 1],
    }


def point_metrics(y, pred, prob=None):
    out = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "flagged": int(pred.sum()),
        "false_alarms": int(((pred == 1) & (y == 0)).sum()),
        "missed": int(((pred == 0) & (y == 1)).sum()),
    }
    if prob is not None:
        out["average_precision"] = average_precision_score(y, prob)
    return out


def hard(prob):
    """The model's yes/no decision. Strictly greater than 0.5, which is exactly what
    sklearn's predict() does for a binary classifier (a tie goes to class 0)."""
    return (prob > 0.5).astype(int)


def metrics_table(y, probs, rule_pred):
    """One row per model: hard-decision metrics, plus average precision."""
    rows = {RULE: point_metrics(y, rule_pred)}
    for name, prob in probs.items():
        rows[name] = point_metrics(y, hard(prob), prob)
    return pd.DataFrame(rows).T[
        ["precision", "recall", "f1", "average_precision", "flagged", "false_alarms", "missed"]
    ]


def pr_curve(y, prob):
    precision, recall, _ = precision_recall_curve(y, prob)
    return recall, precision, average_precision_score(y, prob)


def paired_bootstrap(y, pred_a, pred_b, n=N_BOOTSTRAP, seed=SEED):
    """
    Resample invoices with replacement; score both models on each identical resample.
    Returns the per-resample metrics for both models plus the F1 difference (b - a).
    """
    rng = np.random.default_rng(seed)
    size = len(y)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        yy = y[idx]
        a, b = pred_a[idx], pred_b[idx]
        row = {}
        for label, pred in (("a", a), ("b", b)):
            row[f"{label}_precision"] = precision_score(yy, pred, zero_division=0)
            row[f"{label}_recall"] = recall_score(yy, pred, zero_division=0)
            row[f"{label}_f1"] = f1_score(yy, pred, zero_division=0)
        row["f1_diff"] = row["b_f1"] - row["a_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def interval(samples, level=0.95):
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def uncertainty_table(y, pred_a, pred_b, name_a, name_b):
    """Point estimate and 95% bootstrap interval for each metric of both models."""
    boot = paired_bootstrap(y, pred_a, pred_b)
    rows = []
    for key, name, pred in (("a", name_a, pred_a), ("b", name_b, pred_b)):
        for metric, fn in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            lo, hi = interval(boot[f"{key}_{metric}"])
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": fn(y, pred, zero_division=0),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    diff_lo, diff_hi = interval(boot["f1_diff"])
    diff = {
        "estimate": f1_score(y, pred_b) - f1_score(y, pred_a),
        "ci_low": diff_lo,
        "ci_high": diff_hi,
        "share_of_resamples_b_better": float((boot["f1_diff"] > 0).mean()),
    }
    return pd.DataFrame(rows), diff


def wilson(flags, n, z=1.96):
    """Wilson score interval for a proportion -- behaves at small n and near zero,
    where the textbook normal approximation produces negative rates."""
    if n == 0:
        return float("nan"), float("nan")
    p = flags / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def fairness_table(test, pred_shipped, pred_naive):
    """False-alarm rate on truly clean invoices, per vendor region, with 95% intervals."""
    rows = []
    for region in sorted(test["region"].unique()):
        mask = ((test["region"] == region) & (test["true_fraud"] == 0)).to_numpy()
        n = int(mask.sum())
        for model, pred in ((TREE, pred_shipped), (NAIVE, pred_naive)):
            flags = int(pred[mask].sum())
            lo, hi = wilson(flags, n)
            rows.append(
                {
                    "region": region,
                    "model": model,
                    "clean_invoices": n,
                    "false_alarms": flags,
                    "rate": flags / n,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def importance_table(bundle, test):
    """Permutation importance on the held-out test set, scored by average precision,
    measured identically for every model so the rankings are comparable."""
    X, y = test[FEATURE_NAMES], test["true_fraud"].to_numpy()
    frames = []
    for name, model in ((TREE, bundle["responsible"]), (FOREST, bundle["forest"]), (LOGISTIC, bundle["logistic"])):
        result = permutation_importance(
            model,
            X,
            y,
            scoring="average_precision",
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=SEED,
            n_jobs=-1,
        )
        frames.append(
            pd.DataFrame(
                {
                    "feature": FEATURE_NAMES,
                    "model": name,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def routing_table(test, prob_shipped):
    """What the agent does to the payment run: how many invoices land in each queue,
    and how much of the fraud each queue contains."""
    action = np.where(
        prob_shipped >= ESCALATE_ABOVE,
        "ESCALATE",
        np.where(prob_shipped >= RELEASE_BELOW, "HOLD", "RELEASE"),
    )
    fraud = test["true_fraud"].to_numpy()
    frame = pd.DataFrame(
        {
            "action": action,
            "fraud": fraud,
            "fraud_amount": test["amount"].to_numpy() * fraud,
        }
    )
    out = frame.groupby("action").agg(
        invoices=("fraud", "size"),
        fraudulent=("fraud", "sum"),
        fraud_dollars=("fraud_amount", "sum"),
    )
    out = out.reindex(["ESCALATE", "HOLD", "RELEASE"]).fillna(0)
    out["share_of_queue"] = out["invoices"] / out["invoices"].sum()
    out["share_of_fraud"] = out["fraudulent"] / max(out["fraudulent"].sum(), 1)
    return out
