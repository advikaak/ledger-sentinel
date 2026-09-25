"""
The agent.

A classifier returns 0.83. An agent decides what to do about it and writes down why.
Four stages:

  PERCEIVE  pull the invoices waiting for payment approval
  DECIDE    score each one, and recover the exact rule chain the tree followed
  ACT       route it -- release for payment, hold for review, or escalate
  EXPLAIN   turn the evidence into a memo an AP analyst can act on

The explanation is generated from the model's own decision path plus the specific
records involved -- the invoice it matched, the bank change that preceded it. The
language model narrates evidence that already exists. It is never asked whether the
invoice is fraudulent, because that is the model's job and an LLM guessing at it
would be the least reliable part of the system.
"""

import os
import pickle

import pandas as pd
from sklearn.tree import _tree

from features import FEATURE_LABELS, FEATURE_NAMES

RELEASE_BELOW = 0.35
ESCALATE_ABOVE = 0.75

MODEL = "claude-sonnet-5"


def decision_path(tree, feature_names, x_row):
    """
    Recover the literal sequence of tests this invoice failed or passed.

    This is the whole argument for shipping a decision tree. Not "similarity was an
    important feature in general" -- but "for THIS invoice, the model asked three
    questions in this order and got these answers." An analyst can check each one by
    hand, and a vendor disputing a held payment can be told exactly what triggered it.
    """
    inner = tree.tree_
    node, steps = 0, []
    while inner.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_names[inner.feature[node]]
        threshold = inner.threshold[node]
        value = float(x_row[name])
        went_left = value <= threshold
        steps.append(
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "value": value,
                "threshold": float(threshold),
                "test": f"{name} {'<=' if went_left else '>'} {threshold:.2f}",
            }
        )
        node = inner.children_left[node] if went_left else inner.children_right[node]
    return steps


def gather_evidence(invoice, ledger, changes):
    """
    Pull the actual records behind the score. A memo that says "similarity 0.99" is
    useless to an analyst; a memo that says "this matches INV-40231, paid on day 412,
    never credited" is something they can open and check.
    """
    evidence = {}
    prior = ledger[
        (ledger.vendor_id == invoice["vendor_id"]) & (ledger.day < invoice["day"])
    ]

    if len(prior):
        from difflib import SequenceMatcher

        def norm(desc, iid):
            return f"{'-'.join(str(iid).split('-')[:2])} {desc}".lower()

        this_text = norm(invoice["description"], invoice["invoice_id"])
        scored = [
            (SequenceMatcher(None, this_text, norm(r.description, r.invoice_id)).ratio(), r)
            for r in prior.itertuples()
        ]
        similarity, match = max(scored, key=lambda kv: kv[0])
        if similarity > 0.80:
            evidence["closest_prior_invoice"] = {
                "invoice_id": match.invoice_id,
                "amount": float(match.amount),
                "day": int(match.day),
                "days_before": int(invoice["day"] - match.day),
                "similarity": round(similarity, 3),
                "was_credited_back": bool(match.cancelled),
                "amount_difference": round(float(invoice["amount"]) - float(match.amount), 2),
            }

    vendor_changes = changes[
        (changes.vendor_id == invoice["vendor_id"]) & (changes.change_day <= invoice["day"])
    ]
    if len(vendor_changes):
        last = vendor_changes.sort_values("change_day").iloc[-1]
        evidence["bank_detail_change"] = {
            "changed_on_day": int(last.change_day),
            "days_before_this_invoice": int(invoice["day"] - last.change_day),
            "old_account": f"****{str(last.old_account)[-4:]}",
            "new_account": f"****{str(last.new_account)[-4:]}",
        }

    sharing = ledger[
        (ledger.bank_account == invoice["bank_account"])
        & (ledger.vendor_id != invoice["vendor_id"])
        & (ledger.day <= invoice["day"])
    ]["vendor_id"].unique()
    if len(sharing):
        evidence["bank_account_shared_with"] = sorted(sharing.tolist())

    vendor_history = ledger[
        (ledger.vendor_id == invoice["vendor_id"]) & (ledger.day < invoice["day"])
    ]
    evidence["vendor_profile"] = {
        "prior_invoices": int(len(vendor_history)),
        "median_prior_amount": (
            round(float(vendor_history["amount"].median()), 2) if len(vendor_history) else None
        ),
        "this_amount": float(invoice["amount"]),
        "purchase_order": invoice["po_number"] or "NONE",
    }
    return evidence


def route(probability):
    """What the agent DOES. The thresholds encode a policy, not a fact -- a company
    with a small AP team would raise them to keep the review queue manageable."""
    if probability >= ESCALATE_ABOVE:
        return "ESCALATE", "block payment, notify treasury, confirm with vendor out of band"
    if probability >= RELEASE_BELOW:
        return "HOLD", "hold for analyst review before the payment run"
    return "RELEASE", "release into the payment run"


def build_case(invoice, features_row, model, ledger, changes):
    probability = float(model.predict_proba(features_row[FEATURE_NAMES].to_frame().T)[0, 1])
    action, instruction = route(probability)
    return {
        "invoice_id": invoice["invoice_id"],
        "vendor_id": invoice["vendor_id"],
        "amount": float(invoice["amount"]),
        "risk_score": round(probability, 3),
        "action": action,
        "instruction": instruction,
        "decision_path": decision_path(model, FEATURE_NAMES, features_row),
        "evidence": gather_evidence(invoice, ledger, changes),
    }


SYSTEM = """You write case notes for an accounts-payable fraud analyst.

You are given a completed risk assessment: a score from a decision-tree model, the \
exact rule chain that produced it, and the underlying records. Your job is to turn \
that into a short memo the analyst can act on.

Rules:
- Never re-judge the invoice. The score is the model's decision, not yours.
- Cite the specific records -- invoice numbers, day numbers, amounts, account suffixes.
- State the benign explanation as well when the evidence permits one. Most flagged \
invoices are legitimate, and an analyst who is only ever shown the incriminating \
reading stops reading.
- Give the analyst a concrete first check to run.
- Four sentences maximum. No headings, no bullet points, no preamble."""


def explain(case):
    """
    Narrate one case. Requires ANTHROPIC_API_KEY; there is no offline path, and a
    missing key fails loudly rather than degrading silently.
    """
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. The agent's explanation stage calls Claude; "
            "export the key before running the demo."
        )

    path = "\n".join(
        f"  - {s['label']}: measured {s['value']:.2f}, rule was {s['test']}"
        for s in case["decision_path"]
    )
    import json

    prompt = (
        f"Invoice {case['invoice_id']} from vendor {case['vendor_id']}, "
        f"${case['amount']:,.2f}.\n"
        f"Model risk score: {case['risk_score']:.2f} -> {case['action']}\n\n"
        f"Rule chain the model followed:\n{path}\n\n"
        f"Supporting records:\n{json.dumps(case['evidence'], indent=2, default=str)}"
    )

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def load_agent():
    bundle = pickle.load(open("models/models.pkl", "rb"))
    ledger = pd.read_csv("data/invoices.csv").fillna({"po_number": "", "fraud_type": ""})
    changes = pd.read_csv("data/bank_changes.csv")
    return bundle["responsible"], ledger, changes


def run_queue(invoice_ids, narrate=True):
    """PERCEIVE -> DECIDE -> ACT -> EXPLAIN over a batch of pending invoices."""
    model, ledger, changes = load_agent()
    scored = pd.read_csv("data/test.csv").fillna({"po_number": "", "fraud_type": ""})

    cases = []
    for invoice_id in invoice_ids:
        row = scored[scored.invoice_id == invoice_id]
        if row.empty:
            raise KeyError(f"{invoice_id} is not in the pending queue")
        row = row.iloc[0]
        case = build_case(row, row, model, ledger, changes)
        if narrate:
            case["memo"] = explain(case)
        cases.append(case)
    return cases


if __name__ == "__main__":
    import json
    import sys

    ids = sys.argv[1:]
    if not ids:
        raise SystemExit("usage: python src/agent.py INV-XXXXX [INV-YYYYY ...]")
    print(json.dumps(run_queue(ids), indent=2, default=str))
