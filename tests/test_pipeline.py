"""
Checks that the claims in the README hold: no leakage, an honest time split, the
sensitive column really is excluded, the agent's demo decisions are right, and the
headline numbers reproduce exactly.
"""

import json
import pickle

import pandas as pd
import pytest

import agent
from features import FEATURE_NAMES, build_feature_table
from generate_data import build_ledger


# ---------------------------------------------------------------- data + features

def test_ledger_is_reproducible():
    first = build_ledger()
    second = build_ledger()
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_features_use_only_the_past():
    """
    Point-in-time correctness: an invoice's features must not change when invoices
    from its future are added to the ledger. If they did, the model would be
    trained on information it could never have at payment time.
    """
    vendors, invoices, changes = build_ledger()
    cutoff = 400
    full = build_feature_table(invoices, vendors, changes).set_index("invoice_id")
    # Truncate BOTH sources of history: future invoices and future bank changes.
    past_only = build_feature_table(
        invoices[invoices["day"] < cutoff], vendors, changes[changes["change_day"] < cutoff]
    ).set_index("invoice_id")

    assert len(past_only) > 1000
    pd.testing.assert_frame_equal(past_only, full.loc[past_only.index])


def test_no_labels_or_identity_in_features():
    forbidden = {"true_fraud", "historical_label", "fraud_type", "region", "vendor_id", "name"}
    assert forbidden.isdisjoint(FEATURE_NAMES)


# ---------------------------------------------------------------- split + models

def test_time_split_is_ordered_and_disjoint(artifacts):
    fit = pd.read_csv("data/train.csv")          # train + validation, refit after tuning
    validation = pd.read_csv("data/validation.csv")
    test = pd.read_csv("data/test.csv")
    bundle = pickle.load(open("models/models.pkl", "rb"))

    train_only = fit[fit["day"] < bundle["train_cut"]]
    assert train_only["day"].max() < validation["day"].min()
    assert validation["day"].max() < test["day"].min()
    assert set(fit["invoice_id"]).isdisjoint(test["invoice_id"])


def test_shipped_model_never_sees_region(artifacts):
    bundle = pickle.load(open("models/models.pkl", "rb"))
    shipped = bundle["responsible"]
    assert list(shipped.feature_names_in_) == FEATURE_NAMES
    assert any(name.startswith("region=") for name in bundle["naive"].feature_names_in_)


def test_headline_numbers_reproduce(artifacts):
    """
    Pins the numbers quoted in the README and notebook to the code. If this fails
    after a deliberate change to the generator or the model, update the write-ups.
    """
    results = json.load(open("out/evaluation.json"))["performance"]
    shipped = results["decision tree (shipped)"]
    assert round(shipped["precision"], 3) == 0.560
    assert round(shipped["recall"], 3) == 0.903
    assert results["keyword rule (no ML)"]["recall"] == pytest.approx(5 / 31)

    fairness = json.load(open("out/evaluation.json"))["fairness"]["South Asia"]
    assert round(fairness["shipped"], 3) == 0.160
    assert round(fairness["naive"], 3) == 0.617


# ---------------------------------------------------------------- agent

def test_demo_cases_are_decided_correctly(artifacts):
    import demo

    model, ledger, changes = agent.load_agent()
    test = pd.read_csv("data/test.csv").fillna({"po_number": "", "fraud_type": ""})
    picks = demo.pick_cases(test, model, ledger, changes)

    assert [label for label, _, _ in picks] == [
        "VENDOR IMPERSONATION",
        "DUPLICATE PAYMENT",
        "LOOKS GUILTY, ISN'T",
    ]
    for label, case, record in picks:
        flagged = case["action"] != "RELEASE"
        assert flagged == bool(record["true_fraud"]), label
        assert case["decision_path"], "every decision must come with its rule chain"


def test_explain_fails_loudly_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        agent.explain({"invoice_id": "X"})


def test_explain_narrates_evidence_without_asking_for_a_verdict(monkeypatch):
    """The language model receives the model's decision and the records, and is told
    not to re-judge. This checks the request without calling the network."""
    import anthropic

    sent = {}

    class FakeMessages:
        def create(self, **kwargs):
            sent.update(kwargs)

            class Block:
                text = "  memo text  "

            class Response:
                content = [Block()]

            return Response()

    class FakeClient:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)

    case = {
        "invoice_id": "INV-1",
        "vendor_id": "V1",
        "amount": 1000.0,
        "risk_score": 0.9,
        "action": "ESCALATE",
        "decision_path": [{"label": "days since bank change", "value": 5.0, "test": "days_since_bank_change <= 10.5"}],
        "evidence": {"bank_detail_change": {"days_before_this_invoice": 5}},
    }
    memo = agent.explain(case)

    assert memo == "memo text"
    assert sent["model"] == agent.MODEL
    assert "Never re-judge" in sent["system"]
    user_text = sent["messages"][0]["content"]
    assert "INV-1" in user_text and "days_before_this_invoice" in user_text
