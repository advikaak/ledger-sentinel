"""
Robustness checks: how far do the headline numbers move when the world is less kind?

evaluate.py reports one ledger, one seed, a 5% fraud rate and signals we planted
ourselves. This module asks the four questions a skeptical reviewer would ask:

  1. SEEDS      Is 90% recall a property of the method or of one lucky draw of 31 fraud
                cases? Rebuild the whole world from ten different seeds.
  2. PREVALENCE Real fraud is far rarer than 5%. What happens to precision, the number
                an analyst feels, when it is 1% or 0.1%?
  3. SIGNAL     We planted the timing tell and the credit-note control. How much do the
                results degrade when those signals are weaker?
  4. ADAPTIVE   What if the fraudster knows the model watches billing rhythm and waits?

It also re-tunes the agent's HOLD band on validation cost (question 5), because the
shipped thresholds were set by policy and the hold band caught no fraud on test.

Everything here uses the same features, the same shipped-model settings, the same
time-ordered split, and the same hard 0.5 decision as the rest of the project. The
shipped model and the numbers in README.md are untouched. Output: out/robustness.json.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from analysis import keyword_rule
from features import FEATURE_NAMES, build_feature_table
from generate_data import build_ledger
from train import COST_OF_MISS_VS_FALSE_ALARM, SEED, time_split, tune

SEEDS = [20261113 + 7 * k for k in range(10)]  # the project's own seed is the first
SWEEP_SEEDS = SEEDS[:5]
WEIGHT = {0: 1, 1: COST_OF_MISS_VS_FALSE_ALARM}


# --------------------------------------------------------------------------
# One world, end to end
# --------------------------------------------------------------------------
def make_world(seed, **ledger_kwargs):
    """Ledger plus point-in-time features, in the same shape train.load() returns."""
    vendors, invoices, changes = build_ledger(seed, **ledger_kwargs)
    invoices = invoices.fillna({"po_number": "", "fraud_type": ""})
    features = build_feature_table(invoices, vendors, changes)
    data = (
        invoices.merge(features, on="invoice_id")
        .merge(vendors[["vendor_id", "region"]], on="vendor_id")
        .sort_values(["day", "invoice_id"])
        .reset_index(drop=True)
    )
    return data


def fit_tree(fit_data, params):
    return DecisionTreeClassifier(class_weight=WEIGHT, random_state=SEED, **params).fit(
        fit_data[FEATURE_NAMES], fit_data["historical_label"]
    )


def fit_forest(fit_data):
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight=WEIGHT, random_state=SEED, n_jobs=-1
    ).fit(fit_data[FEATURE_NAMES], fit_data["historical_label"])


def train_world(data):
    """The project's recipe: tune on validation, refit on train + validation."""
    train, validation, test, train_cut, val_cut = time_split(data)
    params, _ = tune(train, validation, WEIGHT)
    fit_data = pd.concat([train, validation], ignore_index=True)
    return {
        "tree": fit_tree(fit_data, params),
        "forest": fit_forest(fit_data),
        "params": params,
        "train": train,
        "validation": validation,
        "test": test,
        "val_cut": val_cut,
    }


def confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def summarize(y, pred):
    c = confusion(y, pred)
    flagged = c["tp"] + c["fp"]
    return {
        **c,
        "precision": c["tp"] / flagged if flagged else 0.0,
        "recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0,
    }


def hard(model, df):
    """Strictly greater than 0.5, which is what predict() does for a binary tree."""
    return (model.predict_proba(df[FEATURE_NAMES])[:, 1] > 0.5).astype(int)


def spread(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "per_seed": [round(float(v), 3) for v in values],
    }


# --------------------------------------------------------------------------
# 1. Seeds
# --------------------------------------------------------------------------
def run_seeds():
    rows, pooled = [], {"tree": [], "rule": []}
    for seed in SEEDS:
        world = train_world(make_world(seed))
        test = world["test"]
        y = test["true_fraud"].to_numpy()
        tree = summarize(y, hard(world["tree"], test))
        forest = summarize(y, hard(world["forest"], test))
        rule = summarize(y, keyword_rule(test))
        rows.append(
            {
                "seed": seed,
                "test_fraud": int(y.sum()),
                "tree": tree,
                "forest": forest,
                "rule": rule,
                "tree_depth": world["params"]["max_depth"],
            }
        )
        pooled["tree"].append(tree)
        pooled["rule"].append(rule)
        print(
            f"  seed {seed}: tree recall {tree['recall']:.2f} precision {tree['precision']:.2f}"
            f" | forest recall {forest['recall']:.2f} precision {forest['precision']:.2f}"
            f" | rule recall {rule['recall']:.2f} ({int(y.sum())} fraud in test)"
        )

    def col(model, key):
        return [r[model][key] for r in rows]

    out = {
        "seeds": rows,
        "tree_recall": spread(col("tree", "recall")),
        "tree_precision": spread(col("tree", "precision")),
        "forest_recall": spread(col("forest", "recall")),
        "forest_precision": spread(col("forest", "precision")),
        "rule_recall": spread(col("rule", "recall")),
        "rule_precision": spread(col("rule", "precision")),
    }
    # Pooling counts across seeds gives one larger test sample for later steps.
    for model in ("tree", "forest", "rule"):
        out[f"{model}_pooled"] = {
            k: int(sum(r[model][k] for r in rows)) for k in ("tp", "fn", "fp", "tn")
        }
    return out


# --------------------------------------------------------------------------
# 2. Prevalence
# --------------------------------------------------------------------------
def precision_at_prevalence(recall, false_alarm_rate, prevalence):
    """
    Precision if fraud made up `prevalence` of invoices, holding the model's recall and
    its false-alarm rate on clean invoices fixed. Bayes' rule, nothing fitted.

    Assumption, stated plainly: the model behaves the same on each fraud and each clean
    invoice when fraud is rarer. It was also trained at ~5% fraud; a model trained on
    rarer fraud would have fewer examples to learn from, so this is, if anything, kind.
    """
    tp = recall * prevalence
    fp = false_alarm_rate * (1 - prevalence)
    return tp / (tp + fp) if (tp + fp) else 0.0


def run_prevalence(pooled):
    out = {}
    for model, c in pooled.items():
        recall = c["tp"] / (c["tp"] + c["fn"])
        far = c["fp"] / (c["fp"] + c["tn"])
        out[model] = {
            "recall": recall,
            "false_alarm_rate": far,
            "precision_at": {
                f"{p:.3%}": precision_at_prevalence(recall, far, p)
                for p in (0.05, 0.01, 0.005, 0.001)
            },
        }
    return out


# --------------------------------------------------------------------------
# 3. Weaker planted signals
# --------------------------------------------------------------------------
def duplicate_recall(test, pred):
    mask = (test["fraud_type"] == "duplicate_payment").to_numpy()
    return float(pred[mask].mean()) if mask.any() else float("nan")


def run_signal_sweep():
    settings = [
        ("as shipped", {}),
        ("careless duplicates 35%", {"careless_share": 0.35}),
        ("careless duplicates 10%", {"careless_share": 0.10}),
        ("credit notes recorded 50%", {"credit_rate": 0.50}),
        ("both weaker", {"careless_share": 0.10, "credit_rate": 0.50}),
    ]
    out = []
    for label, kwargs in settings:
        overall_r, overall_p, dup_r = [], [], []
        for seed in SWEEP_SEEDS:
            world = train_world(make_world(seed, **kwargs))
            test = world["test"]
            pred = hard(world["tree"], test)
            s = summarize(test["true_fraud"].to_numpy(), pred)
            overall_r.append(s["recall"])
            overall_p.append(s["precision"])
            dup_r.append(duplicate_recall(test, pred))
        out.append(
            {
                "setting": label,
                "recall": spread(overall_r),
                "precision": spread(overall_p),
                "duplicate_recall": spread(dup_r),
            }
        )
        print(
            f"  {label:28s} recall {np.mean(overall_r):.2f}  precision {np.mean(overall_p):.2f}"
            f"  duplicate recall {np.nanmean(dup_r):.2f}"
        )
    return out


# --------------------------------------------------------------------------
# 4. The adaptive fraudster
# --------------------------------------------------------------------------
def run_adaptive():
    """
    Same seed, same honest invoices, same bank-swap and phantom-vendor fraud. Only the
    duplicate-payment fraudsters change: they wait one full billing cycle and let the
    amount drift like an honest recurring invoice.

    A. attacked:  model trained on the ordinary world, scored on the adaptive world.
    B. retrained: model trained on the adaptive world (the defender has seen the
                  attack in labelled history), scored on the adaptive world.
    """
    out = {"attacked": [], "retrained": [], "unattacked": []}
    for seed in SWEEP_SEEDS:
        normal = train_world(make_world(seed))
        adaptive_data = make_world(seed, adaptive=True)
        # Score the adaptive world on the same calendar period as the ordinary test.
        adaptive_test = adaptive_data[adaptive_data["day"] >= normal["val_cut"]].reset_index(drop=True)
        y = adaptive_test["true_fraud"].to_numpy()

        pred_attacked = hard(normal["tree"], adaptive_test)
        retrained = train_world(adaptive_data)
        pred_retrained = hard(retrained["tree"], adaptive_test)
        pred_normal = hard(normal["tree"], normal["test"])

        for key, df, pred in (
            ("unattacked", normal["test"], pred_normal),
            ("attacked", adaptive_test, pred_attacked),
            ("retrained", adaptive_test, pred_retrained),
        ):
            s = summarize(df["true_fraud"].to_numpy(), pred)
            s["duplicate_recall"] = duplicate_recall(df, pred)
            s["duplicates_in_test"] = int((df["fraud_type"] == "duplicate_payment").sum())
            out[key].append(s)
        print(
            f"  seed {seed}: duplicate recall unattacked {out['unattacked'][-1]['duplicate_recall']:.2f}"
            f" -> attacked {out['attacked'][-1]['duplicate_recall']:.2f}"
            f" -> retrained {out['retrained'][-1]['duplicate_recall']:.2f}"
        )

    summary = {}
    for key, rows in out.items():
        summary[key] = {
            "duplicate_recall": spread([r["duplicate_recall"] for r in rows]),
            "overall_recall": spread([r["recall"] for r in rows]),
            "overall_precision": spread([r["precision"] for r in rows]),
            "duplicates_in_test_total": int(sum(r["duplicates_in_test"] for r in rows)),
        }
    return summary


# --------------------------------------------------------------------------
# 5. Re-tuning the hold band on validation cost
# --------------------------------------------------------------------------
# Review costs, in the same unit the project already uses (a false alarm = 1). These
# are ASSUMPTIONS, not measurements: escalating blocks a supplier's payment and needs
# an out-of-band call, so we price it at twice a hold. A missed fraud keeps the
# project's existing 6:1 ratio. Change them and the chosen thresholds will move.
COST_MISS = float(COST_OF_MISS_VS_FALSE_ALARM)
COST_HOLD = 1.0
COST_ESCALATE = 2.0


def routing_cost(prob, y, lo, hi):
    """Cost of a (hold, escalate) threshold pair: reviews cost, missed fraud costs."""
    escalate = prob >= hi
    hold = (prob >= lo) & ~escalate
    released = ~(escalate | hold)
    return (
        COST_ESCALATE * escalate.sum()
        + COST_HOLD * hold.sum()
        + COST_MISS * ((y == 1) & released).sum()
    )


def routing_table(prob, y, amount, lo, hi):
    rows = {}
    for name, mask in (
        ("escalate", prob >= hi),
        ("hold", (prob >= lo) & (prob < hi)),
        ("release", prob < lo),
    ):
        rows[name] = {
            "invoices": int(mask.sum()),
            "fraudulent": int(y[mask].sum()),
            "fraud_dollars": float(amount[mask & (y == 1)].sum()),
        }
    return rows


def run_hold_band():
    """
    Fit on train only so validation probabilities are honest, grid-search both
    thresholds on validation cost, then score the sealed test slice once. The shipped
    model is not refit here; thresholds are a policy layer on top of it.
    """
    data = make_world(SEEDS[0])
    train, validation, test, _, _ = time_split(data)
    params, _ = tune(train, validation, WEIGHT)
    model = fit_tree(train, params)

    pv = model.predict_proba(validation[FEATURE_NAMES])[:, 1]
    yv = validation["true_fraud"].to_numpy()
    grid = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    best, best_cost = None, float("inf")
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            cost = routing_cost(pv, yv, lo, hi)
            if cost < best_cost:
                best, best_cost = (lo, hi), cost

    pt = model.predict_proba(test[FEATURE_NAMES])[:, 1]
    yt = test["true_fraud"].to_numpy()
    amount = test["amount"].to_numpy()
    shipped = (0.35, 0.75)
    return {
        "assumed_costs": {"miss": COST_MISS, "hold": COST_HOLD, "escalate": COST_ESCALATE},
        "shipped_thresholds": {"hold_from": shipped[0], "escalate_from": shipped[1]},
        "tuned_thresholds": {"hold_from": best[0], "escalate_from": best[1]},
        "validation_cost": {
            "shipped": float(routing_cost(pv, yv, *shipped)),
            "tuned": float(best_cost),
        },
        "test_cost": {
            "shipped": float(routing_cost(pt, yt, *shipped)),
            "tuned": float(routing_cost(pt, yt, *best)),
        },
        "test_routing_shipped": routing_table(pt, yt, amount, *shipped),
        "test_routing_tuned": routing_table(pt, yt, amount, *best),
    }


# --------------------------------------------------------------------------
def main():
    print("1. ten seeds (the whole world rebuilt each time)")
    seeds = run_seeds()
    print()
    print("2. precision if fraud were rarer (pooled over the ten test sets)")
    prevalence = run_prevalence(
        {m: seeds[f"{m}_pooled"] for m in ("tree", "forest", "rule")}
    )
    for model, r in prevalence.items():
        shown = ", ".join(f"{k}: {v:.0%}" for k, v in r["precision_at"].items())
        print(
            f"  {model:6s} recall {r['recall']:.0%}, false alarms on clean "
            f"{r['false_alarm_rate']:.1%} -> precision at fraud rate {shown}"
        )
    print()
    print("3. weaker planted signals (5 seeds each)")
    signal = run_signal_sweep()
    print()
    print("4. a fraudster who waits (5 seeds)")
    adaptive = run_adaptive()
    print()
    print("5. re-tune the hold band on validation cost")
    hold = run_hold_band()
    print(f"  shipped {hold['shipped_thresholds']} tuned {hold['tuned_thresholds']}")
    print(f"  test cost shipped {hold['test_cost']['shipped']:.0f} -> tuned {hold['test_cost']['tuned']:.0f}")
    for label in ("test_routing_shipped", "test_routing_tuned"):
        print(f"  {label}: {hold[label]}")

    json.dump(
        {
            "seeds": seeds,
            "prevalence": prevalence,
            "signal_sweep": signal,
            "adaptive_fraudster": adaptive,
            "hold_band": hold,
        },
        open("out/robustness.json", "w"),
        indent=2,
    )
    print("\nwrote out/robustness.json")


if __name__ == "__main__":
    main()
