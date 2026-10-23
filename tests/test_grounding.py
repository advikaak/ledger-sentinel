"""
The grounding check: a note may only state identifiers and figures its case contains.

No network call is involved. The notes below are written by hand from a real case, so
these tests check the checker, not Claude. Whether Claude's live notes pass is something
to run with a key (see the README); it is not claimed here.
"""

import pandas as pd
import pytest

import agent
from grounding import ungrounded_claims


@pytest.fixture(scope="module")
def duplicate_case(artifacts):
    """A real escalated duplicate-payment case from the sealed test period."""
    test = pd.read_csv("data/test.csv").fillna({"po_number": "", "fraud_type": ""})
    model, ledger, changes = agent.load_agent()
    for row in test[test.fraud_type == "duplicate_payment"].itertuples():
        row = test[test.invoice_id == row.invoice_id].iloc[0]
        case = agent.build_case(row, row, model, ledger, changes)
        if case["action"] == "ESCALATE" and "closest_prior_invoice" in case["evidence"]:
            return case
    pytest.fail("no escalated duplicate case with a matching prior invoice in the test period")


def faithful_note(case):
    m = case["evidence"]["closest_prior_invoice"]
    return (
        f"Invoice {case['invoice_id']} for ${case['amount']:,.2f} from {case['vendor_id']} "
        f"closely matches {m['invoice_id']}, paid {m['days_before']} days earlier for "
        f"${m['amount']:,.2f}, and that invoice was not credited back. "
        f"Check the original payment before releasing."
    )


def test_a_note_built_from_the_evidence_passes(duplicate_case):
    assert ungrounded_claims(faithful_note(duplicate_case), duplicate_case) == []


def test_rounding_is_allowed(duplicate_case):
    m = duplicate_case["evidence"]["closest_prior_invoice"]
    note = f"It matches {m['invoice_id']} for about ${round(m['amount']):,}."
    assert ungrounded_claims(note, duplicate_case) == []


def test_an_invented_invoice_number_is_caught(duplicate_case):
    note = faithful_note(duplicate_case) + " It also resembles INV-99999."
    problems = ungrounded_claims(note, duplicate_case)
    assert any("INV-99999" in p for p in problems)


def test_an_invented_amount_is_caught(duplicate_case):
    note = faithful_note(duplicate_case) + " The vendor was previously paid $987,654 in total."
    problems = ungrounded_claims(note, duplicate_case)
    assert any("987,654" in p for p in problems)


def test_an_invented_account_suffix_is_caught(duplicate_case):
    note = faithful_note(duplicate_case) + " Funds went to account ****0000."
    problems = ungrounded_claims(note, duplicate_case)
    assert any("0000" in p for p in problems)


def test_invented_day_counts_are_caught(duplicate_case):
    m = duplicate_case["evidence"]["closest_prior_invoice"]
    wrong_days = m["days_before"] + 37
    note = f"It arrived {wrong_days} days after {m['invoice_id']}."
    problems = ungrounded_claims(note, duplicate_case)
    assert any(str(wrong_days) in p for p in problems)
