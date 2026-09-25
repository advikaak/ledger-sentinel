"""
Write everything the website shows into docs/data/site-data.js.

The site is static: no server, nothing to keep awake. The trained models run here,
once, over the sealed test period, and the results ship with the page. The data is
written as a script that sets window.LEDGER (rather than a .json file) so the page
also works when opened straight from disk, where browsers block fetch().

Run after train.py and evaluate.py:

    python3 src/export_site.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A
import demo
from agent import ESCALATE_ABOVE, RELEASE_BELOW, build_case, load_agent, route
from describe import describe_case
from features import EXCLUDED_FEATURES, FEATURE_LABELS, FEATURE_NAMES, FEATURE_SHORT

OUT = Path("docs/data/site-data.js")

MODEL_KEYS = {A.TREE: "tree", A.FOREST: "forest", A.LOGISTIC: "logistic", A.NAIVE: "naive"}
SHOWCASE_BUTTONS = {
    "VENDOR IMPERSONATION": "Impersonation caught",
    "DUPLICATE PAYMENT": "Duplicate caught",
    "LOOKS GUILTY, ISN'T": "Looks guilty, isn't",
}


def r(x, digits=4):
    return round(float(x), digits)


def shares(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return [r(c / len(values)) for c in counts]


def main():
    bundle, train, test = A.load()
    model, ledger, changes = load_agent()
    full = pd.concat([train, test], ignore_index=True)
    y = test["true_fraud"].to_numpy()
    probs = A.scored_models(bundle, test)
    rule_pred = A.keyword_rule(test)

    # ------------------------------------------------------------ queue + cases
    queue, cases = [], {}
    order = np.argsort(-probs[A.TREE], kind="stable")
    for i in order:
        record = test.iloc[i]
        case = build_case(record, record, model, ledger, changes)
        cases[case["invoice_id"]] = describe_case(case, record)
        queue.append({
            "id": case["invoice_id"],
            "vendor": case["vendor_id"],
            "amount": r(case["amount"], 2),
            "risk": r(case["risk_score"], 3),
            "decision": route(case["risk_score"])[0],
            "fraud": int(record["true_fraud"]),
        })

    showcase = [
        {"label": SHOWCASE_BUTTONS[label], "id": case["invoice_id"]}
        for label, case, _ in demo.pick_cases(test, model, ledger, changes)
    ]

    # ------------------------------------------------------------ headline numbers
    routing = A.routing_table(test, probs[A.TREE])
    flagged = ["ESCALATE", "HOLD"]
    metrics = A.metrics_table(y, probs, rule_pred)
    kpis = {
        "caught": r(routing.loc[flagged, "fraudulent"].sum() / routing["fraudulent"].sum()),
        "reviewed": r(routing.loc[flagged, "invoices"].sum() / routing["invoices"].sum()),
        "dollars": r(routing.loc[flagged, "fraud_dollars"].sum() / routing["fraud_dollars"].sum()),
        "rule": r(metrics.loc[A.RULE, "recall"]),
    }

    models = []
    for name, row in metrics.iterrows():
        models.append({
            "name": name,
            "key": MODEL_KEYS.get(name, "rule"),
            "precision": r(row["precision"]),
            "recall": r(row["recall"]),
            "f1": r(row["f1"]),
            "ap": None if np.isnan(row["average_precision"]) else r(row["average_precision"]),
            "flagged": int(row["flagged"]),
            "missed": int(row["missed"]),
        })

    unc, gap = A.uncertainty_table(y, A.hard(probs[A.TREE]), A.hard(probs[A.FOREST]), A.TREE, A.FOREST)
    uncertainty = {
        MODEL_KEYS[m]: {
            row["metric"]: [r(row["estimate"]), r(row["ci_low"]), r(row["ci_high"])]
            for _, row in unc[unc["model"] == m].iterrows()
        }
        for m in (A.TREE, A.FOREST)
    }
    uncertainty["f1_gap"] = {k: r(v) for k, v in gap.items()}

    # ------------------------------------------------------------ charts
    pr = []
    for name in (A.TREE, A.FOREST, A.LOGISTIC, A.NAIVE):
        recall, precision, ap = A.pr_curve(y, probs[name])
        pr.append({
            "key": MODEL_KEYS[name],
            "name": name,
            "ap": r(ap),
            "points": [[r(a), r(b)] for a, b in zip(recall, precision)],
        })

    legit = full[full["true_fraud"] == 0]
    fraud = full[full["true_fraud"] == 1]
    sim_edges = np.linspace(0, 1, 41)
    cad_edges = np.linspace(0, 3, 41)
    hist = {
        "similarity": {
            "edges": [r(e) for e in sim_edges],
            "legit": shares(legit["max_similarity_to_prior"], sim_edges),
            "fraud": shares(fraud["max_similarity_to_prior"], sim_edges),
            "threshold": 0.95,
            "legit_above": r((legit["max_similarity_to_prior"] > 0.95).mean()),
        },
        "cadence": {
            "edges": [r(e) for e in cad_edges],
            "legit": shares(legit["gap_vs_vendor_cadence"].clip(upper=3), cad_edges),
            "fraud": shares(fraud["gap_vs_vendor_cadence"].clip(upper=3), cad_edges),
        },
    }

    fair = A.fairness_table(test, A.hard(probs[A.TREE]), A.hard(probs[A.NAIVE]))
    fairness = []
    for region, part in fair.groupby("region"):
        part = part.set_index("model")
        fairness.append({
            "region": region,
            "clean": int(part["clean_invoices"].iloc[0]),
            **{
                MODEL_KEYS[m]: {
                    "rate": r(part.loc[m, "rate"]),
                    "lo": r(part.loc[m, "ci_low"]),
                    "hi": r(part.loc[m, "ci_high"]),
                    "flags": int(part.loc[m, "false_alarms"]),
                }
                for m in (A.TREE, A.NAIVE)
            },
        })
    fairness.sort(key=lambda row: -row["naive"]["rate"])

    imp = A.importance_table(bundle, test).pivot(index="feature", columns="model", values="importance")
    importance = [
        {
            "feature": f,
            "label": FEATURE_SHORT[f],
            "tree": r(imp.loc[f, A.TREE]),
            "forest": r(imp.loc[f, A.FOREST]),
            "logistic": r(imp.loc[f, A.LOGISTIC]),
        }
        for f in imp.sort_values(A.TREE, ascending=False).index
    ]

    clean_all = full[full["true_fraud"] == 0]
    mislabelled = clean_all.groupby("region")["historical_label"].mean()

    data = {
        "meta": {
            "ledger_invoices": int(len(full)),
            "vendors": int(full["vendor_id"].nunique()),
            "fraud_rate": r(full["true_fraud"].mean()),
            "test_invoices": int(len(test)),
            "test_fraud": int(y.sum()),
            "escalate_above": ESCALATE_ABOVE,
            "release_below": RELEASE_BELOW,
            "tree_depth": int(bundle["responsible"].get_depth()),
            "cost_ratio": int(bundle["cost_ratio"]),
            "mislabelled_share": r(mislabelled.max()),
            "mislabelled_region": str(mislabelled.idxmax()),
            "base_rate": r(y.mean()),
        },
        "kpis": kpis,
        "showcase": showcase,
        "queue": queue,
        "cases": cases,
        "models": models,
        "uncertainty": uncertainty,
        "routing": [
            {
                "action": action,
                "invoices": int(row["invoices"]),
                "fraudulent": int(row["fraudulent"]),
                "fraud_dollars": r(row["fraud_dollars"], 2),
            }
            for action, row in routing.iterrows()
        ],
        "pr": pr,
        "operating": {
            "tree": [r(metrics.loc[A.TREE, "recall"]), r(metrics.loc[A.TREE, "precision"])],
            "rule": [r(metrics.loc[A.RULE, "recall"]), r(metrics.loc[A.RULE, "precision"])],
        },
        "hist": hist,
        "fairness": fairness,
        "importance": importance,
        "features": [{"name": f, "label": FEATURE_SHORT[f], "meaning": FEATURE_LABELS[f]} for f in FEATURE_NAMES],
        "excluded": [{"name": k, "reason": v} for k, v in EXCLUDED_FEATURES.items()],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "// Generated by src/export_site.py. Do not edit by hand.\n"
        f"window.LEDGER = {payload};\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB): {len(queue)} invoices, {len(cases)} cases")


if __name__ == "__main__":
    main()
