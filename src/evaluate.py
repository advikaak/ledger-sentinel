"""
Evaluate on three axes separately: performance, fairness, interpretability.

Reporting one accuracy number would hide everything that matters here. A model that
is 97% accurate on a 5% fraud base rate may simply be approving everything. A model
with excellent recall may be getting it by over-flagging one region. And a model that
scores well on both is still unusable if nobody can say why it held a payment.

Everything below is scored against TRUE fraud, while the models were trained on the
biased historical labels -- which is the situation a real compliance team inherits.
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

THRESHOLD = 0.5


def rule(row):
    """The rule an AP team writes before anyone suggests machine learning:
    flag anything that looks almost identical to an invoice we already paid."""
    return int(row["max_similarity_to_prior"] > 0.95 and row["amount_ratio_to_closest_match"] > 0.99)


def scores(y_true, y_pred, y_prob=None):
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "flagged": int(y_pred.sum()),
        "false_alarms": int(((y_pred == 1) & (y_true == 0)).sum()),
        "missed": int(((y_pred == 0) & (y_true == 1)).sum()),
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return out


def main():
    bundle = pickle.load(open("models/models.pkl", "rb"))
    feature_names = bundle["feature_names"]
    region_cols = bundle["region_cols"]
    test = pd.read_csv("data/test.csv").fillna({"fraud_type": ""})

    X = test[feature_names]
    y = test["true_fraud"].to_numpy()

    def with_region(df):
        out = df[feature_names].copy()
        for r in region_cols:
            out[f"region={r}"] = (df["region"] == r).astype(float)
        return out

    preds = {
        "keyword rule (no ML)": (test.apply(rule, axis=1).to_numpy(), None),
        "decision tree (shipped)": (
            bundle["responsible"].predict(X),
            bundle["responsible"].predict_proba(X)[:, 1],
        ),
        "logistic regression": (
            bundle["logistic"].predict(X),
            bundle["logistic"].predict_proba(X)[:, 1],
        ),
        "random forest": (
            bundle["forest"].predict(X),
            bundle["forest"].predict_proba(X)[:, 1],
        ),
        "decision tree + region": (
            bundle["naive"].predict(with_region(test)),
            bundle["naive"].predict_proba(with_region(test))[:, 1],
        ),
    }

    # ---------------------------------------------------------------- PERFORMANCE
    print("=" * 74)
    print("AXIS 1 - PERFORMANCE   (test set = the final 20% of the ledger, by time)")
    print("=" * 74)
    print(f"{len(test)} invoices, {int(y.sum())} of them actually fraudulent\n")
    print(f"{'model':<26}{'precision':>10}{'recall':>9}{'F1':>7}{'flags':>7}{'misses':>8}")
    print("-" * 74)
    results = {}
    for name, (pred, prob) in preds.items():
        s = scores(y, pred, prob)
        results[name] = s
        print(
            f"{name:<26}{s['precision']:>9.0%}{s['recall']:>9.0%}"
            f"{s['f1']:>7.2f}{s['flagged']:>7}{s['missed']:>8}"
        )

    print()
    print("Recall matters more than precision here: a false alarm costs an analyst")
    print("ten minutes, a missed fraud costs the invoice amount.")

    # per fraud type -- does the shipped model cover all three schemes?
    shipped = preds["decision tree (shipped)"][0]
    print("\nshipped model, caught by scheme:")
    for kind in ["duplicate_payment", "bank_swap", "phantom_vendor"]:
        mask = (test["fraud_type"] == kind).to_numpy()
        if mask.sum():
            print(f"  {kind:<20}{int(shipped[mask].sum()):>3} of {int(mask.sum()):>3} caught")

    # ------------------------------------------------------------------- FAIRNESS
    print()
    print("=" * 74)
    print("AXIS 2 - FAIRNESS      (does it flag by behavior, or by geography?)")
    print("=" * 74)
    print("Both models trained on the same biased historical labels. The only")
    print("difference is that one was allowed to see vendor region.\n")

    fairness = {}
    print(f"{'region':<18}{'clean invoices':>15}{'flagged by shipped':>20}{'flagged by +region':>20}")
    print("-" * 74)
    for region in region_cols:
        mask = ((test["region"] == region) & (test["true_fraud"] == 0)).to_numpy()
        if mask.sum() < 5:
            continue
        shipped_rate = shipped[mask].mean()
        naive_rate = preds["decision tree + region"][0][mask].mean()
        fairness[region] = {"shipped": float(shipped_rate), "naive": float(naive_rate)}
        print(f"{region:<18}{int(mask.sum()):>15}{shipped_rate:>19.1%}{naive_rate:>19.1%}")

    spread_shipped = max(v["shipped"] for v in fairness.values()) - min(
        v["shipped"] for v in fairness.values()
    )
    spread_naive = max(v["naive"] for v in fairness.values()) - min(
        v["naive"] for v in fairness.values()
    )
    print("-" * 74)
    print(f"{'gap widest-to-narrowest':<33}{spread_shipped:>19.1%}{spread_naive:>19.1%}")
    print()
    print("These are false-alarm rates on INNOCENT vendors. The gap is how much more")
    print("likely a clean vendor is to be held up because of where they are based.")
    print()
    print("The finding that matters is the one that is inconvenient: dropping the region")
    print(f"feature cut the disparity from {spread_naive:.0%} to {spread_shipped:.0%}, but did NOT remove it.")
    print("The biased labels are still in the training data, and the model reaches some")
    print("of the same conclusions through behavioral features that correlate with the")
    print("over-scrutinized group. Deleting a sensitive column is a first step, not a fix;")
    print("the labels themselves would have to be re-adjudicated.")

    # --------------------------------------------------------- INTERPRETABILITY
    print()
    print("=" * 74)
    print("AXIS 3 - INTERPRETABILITY")
    print("=" * 74)
    tree = bundle["responsible"]
    n_leaves = tree.get_n_leaves()
    print(f"shipped model:        decision tree, depth {tree.get_depth()}, {n_leaves} leaves")
    print(f"                      every decision traces to <= {tree.get_depth()} readable conditions")
    tree_r, forest_r = results["decision tree (shipped)"], results["random forest"]
    print(f"                      F1 {tree_r['f1']:.2f}, precision {tree_r['precision']:.0%}, recall {tree_r['recall']:.0%}")
    print(f"random forest:        300 trees, no readable decision path")
    print(f"                      F1 {forest_r['f1']:.2f}, precision {forest_r['precision']:.0%}, recall {forest_r['recall']:.0%}")
    print()
    print("The price of interpretability, stated plainly:")
    print(f"  the forest is {forest_r['f1'] - tree_r['f1']:+.2f} F1 and {forest_r['precision'] - tree_r['precision']:+.0%} precision")
    print(f"  the tree catches {tree_r['recall'] - forest_r['recall']:+.0%} more of the fraud but sends")
    print(f"  {tree_r['false_alarms'] - forest_r['false_alarms']} more clean invoices to an analyst")
    print()
    print("We ship the tree. Holding a supplier's payment is an action that has to be")
    print("justified to that supplier, and 'the model said so' is not a justification.")
    print()
    print("Same data, different model logic, different notion of 'important':")
    tree_imp = dict(zip(feature_names, bundle["responsible"].feature_importances_))
    forest_imp = dict(zip(feature_names, bundle["forest"].feature_importances_))
    logit_imp = dict(
        zip(feature_names, np.abs(bundle["logistic"][-1].coef_[0]))
    )
    print(f"{'':<34}{'tree':>12}{'forest':>12}{'logistic':>12}")
    print("-" * 74)
    for f in sorted(feature_names, key=lambda k: -tree_imp[k])[:6]:
        logit_norm = logit_imp[f] / sum(logit_imp.values())
        print(f"{f:<34}{tree_imp[f]:>11.1%}{forest_imp[f]:>12.1%}{logit_norm:>12.1%}")
    print()
    print("None of them is lying. Feature importance describes how one model used the")
    print("data -- it is not a measure of what causes fraud.")

    json.dump(
        {"performance": results, "fairness": fairness},
        open("out/evaluation.json", "w"),
        indent=2,
        default=float,
    )
    print("\nwritten to out/evaluation.json")


if __name__ == "__main__":
    main()
