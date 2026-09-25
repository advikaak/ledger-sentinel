"""
Fit the models.

Three decisions here are worth defending out loud, because they are the difference
between a project and a demo:

1. WE TRAIN ON THE LABEL A REAL TEAM WOULD HAVE. Not ground truth -- the historical
   investigator write-ups, which in our world carry a regional bias. Then we score
   against ground truth. That gap is the entire fairness story.

2. WE SPLIT BY TIME, NOT AT RANDOM. A random split lets the model train on March to
   predict January. Fraud detection is a forecasting problem; the test set has to be
   the future.

3. WE SHIP A DEPTH-LIMITED DECISION TREE. Not because it scores highest -- the random
   forest does -- but because an AP analyst has to justify holding a supplier's
   payment. A rule chain can be read aloud in a meeting. A forest cannot.

We fit the alternatives anyway and report them, so the trade-off is measured rather
than asserted.
"""

import json
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from features import FEATURE_NAMES, build_feature_table

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20  # the rest is test, and is touched exactly once
SEED = 20261113

# A missed fraud costs the invoice amount. A false alarm costs an analyst ten
# minutes. We tell the model that trade-off explicitly instead of accepting
# whatever balance the class distribution happens to imply.
COST_OF_MISS_VS_FALSE_ALARM = 6

# Three fraud schemes with different signatures means the tree needs enough depth
# to express all three plus the legitimate space. We let validation pick rather than
# asserting a depth, and report whatever it chooses -- including when it wants a
# deeper, less readable tree than we would have preferred.
SEARCH_SPACE = [
    {"max_depth": d, "min_samples_leaf": leaf}
    for d in (3, 4, 5, 6, 7, 8, 10, 12)
    for leaf in (4, 8, 12, 20)
]


def load():
    invoices = pd.read_csv("data/invoices.csv").fillna({"po_number": "", "fraud_type": ""})
    vendors = pd.read_csv("data/vendors.csv")
    changes = pd.read_csv("data/bank_changes.csv")
    features = build_feature_table(invoices, vendors, changes)

    data = (
        invoices.merge(features, on="invoice_id")
        .merge(vendors[["vendor_id", "region"]], on="vendor_id")
        .sort_values(["day", "invoice_id"])
        .reset_index(drop=True)
    )
    return data


def time_split(data):
    """
    Three slices, in time order: train, validation, test.

    The validation slice exists so hyperparameters can be chosen without ever
    consulting the test set. Sweeping settings against test and then reporting the
    best one is how projects end up quoting a number they cannot reproduce in
    production -- the test set has to stay sealed until the end.
    """
    train_cut = int(data.iloc[int(len(data) * TRAIN_FRACTION)]["day"])
    val_cut = int(
        data.iloc[int(len(data) * (TRAIN_FRACTION + VALIDATION_FRACTION))]["day"]
    )
    train = data[data["day"] < train_cut].reset_index(drop=True)
    validation = data[(data["day"] >= train_cut) & (data["day"] < val_cut)].reset_index(drop=True)
    test = data[data["day"] >= val_cut].reset_index(drop=True)
    return train, validation, test, train_cut, val_cut


def tune(train, validation, weight):
    """Pick depth and leaf size on validation, scoring by cost rather than accuracy."""
    best, best_cost = None, float("inf")
    for params in SEARCH_SPACE:
        model = DecisionTreeClassifier(
            class_weight=weight, random_state=SEED, **params
        ).fit(train[FEATURE_NAMES], train["historical_label"])
        pred = model.predict(validation[FEATURE_NAMES])
        truth = validation["historical_label"].to_numpy()
        misses = int(((pred == 0) & (truth == 1)).sum())
        false_alarms = int(((pred == 1) & (truth == 0)).sum())
        cost = misses * COST_OF_MISS_VS_FALSE_ALARM + false_alarms
        if cost < best_cost:
            best, best_cost = params, cost
    return best, best_cost


def main():
    data = load()
    train, validation, test, train_cut, val_cut = time_split(data)
    weight = {0: 1, 1: COST_OF_MISS_VS_FALSE_ALARM}

    print(f"time split at day {train_cut} and day {val_cut}")
    print(f"  train      {len(train):>5} invoices, {int(train.true_fraud.sum()):>3} fraud")
    print(f"  validation {len(validation):>5} invoices, {int(validation.true_fraud.sum()):>3} fraud")
    print(f"  test       {len(test):>5} invoices, {int(test.true_fraud.sum()):>3} fraud  (sealed until evaluate.py)")
    print()

    best_params, best_cost = tune(train, validation, weight)
    print(f"tuned on validation: {best_params}, cost {best_cost}")
    print()

    # Refit on train + validation now that the settings are locked in.
    fit_data = pd.concat([train, validation], ignore_index=True)

    # --- the model we ship: behavior only, shallow enough to read aloud ---
    responsible = DecisionTreeClassifier(
        class_weight=weight, random_state=SEED, **best_params
    ).fit(fit_data[FEATURE_NAMES], fit_data["historical_label"])

    # --- the baseline that does the obvious thing: let the model see region ---
    region_cols = sorted(data["region"].unique())
    naive_features = FEATURE_NAMES + [f"region={r}" for r in region_cols]

    def with_region(df):
        out = df[FEATURE_NAMES].copy()
        for r in region_cols:
            out[f"region={r}"] = (df["region"] == r).astype(float)
        return out

    # Identical settings to the shipped model. The ONLY difference is that this one
    # is allowed to see where the vendor is based.
    naive = DecisionTreeClassifier(
        class_weight=weight, random_state=SEED, **best_params
    ).fit(with_region(fit_data), fit_data["historical_label"])

    # --- alternatives, fit on the same features, to price the trade-off ---
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight=weight,
        random_state=SEED,
        n_jobs=-1,
    ).fit(fit_data[FEATURE_NAMES], fit_data["historical_label"])

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=weight, random_state=SEED),
    ).fit(fit_data[FEATURE_NAMES], fit_data["historical_label"])

    with open("models/models.pkl", "wb") as fh:
        pickle.dump(
            {
                "responsible": responsible,
                "naive": naive,
                "forest": forest,
                "logistic": logistic,
                "feature_names": FEATURE_NAMES,
                "naive_feature_names": naive_features,
                "region_cols": region_cols,
                "best_params": best_params,
                "cost_ratio": COST_OF_MISS_VS_FALSE_ALARM,
                "train_cut": train_cut,
                "val_cut": val_cut,
            },
            fh,
        )

    fit_data.to_csv("data/train.csv", index=False)
    validation.to_csv("data/validation.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    # The shipped model's actual rules, printed. This is the interpretability claim
    # made good -- not a chart of importances, the literal decision logic.
    rules = export_text(responsible, feature_names=FEATURE_NAMES)
    with open("out/decision_rules.txt", "w") as fh:
        fh.write(rules)

    print("shipped model rules (out/decision_rules.txt):")
    print("\n".join(rules.splitlines()[:22]))
    print("   ...")
    print()

    importances = sorted(
        zip(FEATURE_NAMES, responsible.feature_importances_),
        key=lambda kv: -kv[1],
    )
    print("what the shipped tree leaned on:")
    for name, score in importances[:6]:
        if score > 0:
            print(f"  {score:5.1%}  {name}")

    json.dump(
        {"importances": dict(importances), "best_params": best_params},
        open("out/train_summary.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
