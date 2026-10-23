"""
Checks on the machinery behind out/robustness.json (not on its results, which are
measurements, not claims to be asserted).
"""

import pytest

from generate_data import build_ledger
from robustness import precision_at_prevalence


def test_precision_formula_recovers_the_observed_precision():
    # 50 caught of 60 fraud; 40 false alarms among 940 clean invoices. Prevalence 6%.
    recall, false_alarm_rate = 50 / 60, 40 / 940
    observed = 50 / (50 + 40)
    assert precision_at_prevalence(recall, false_alarm_rate, 60 / 1000) == pytest.approx(observed)


def test_rarer_fraud_means_lower_precision():
    values = [precision_at_prevalence(0.9, 0.05, p) for p in (0.05, 0.01, 0.001)]
    assert values[0] > values[1] > values[2]


def test_the_adaptive_attack_changes_only_the_duplicate_payments():
    _, ordinary, _ = build_ledger()
    _, adaptive, _ = build_ledger(adaptive=True)
    key = ["invoice_id", "vendor_id", "amount", "day", "fraud_type"]
    ordinary_rest = ordinary[ordinary.fraud_type != "duplicate_payment"]
    adaptive_rest = adaptive[adaptive.fraud_type != "duplicate_payment"]
    assert ordinary_rest[key].reset_index(drop=True).equals(
        adaptive_rest[key].reset_index(drop=True)
    )
    assert (ordinary.fraud_type == "duplicate_payment").sum() == (
        adaptive.fraud_type == "duplicate_payment"
    ).sum()


def test_defaults_build_the_shipped_ledger():
    default = build_ledger()
    explicit = build_ledger(careless_share=0.65, credit_rate=0.84, adaptive=False)
    for a, b in zip(default, explicit):
        assert a.equals(b)
