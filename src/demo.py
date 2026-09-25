"""
The live demo.

Runs the agent over a queue of pending invoices and prints what it decided and why.
Three cases, chosen to show the system is doing something harder than pattern
matching:

  1. a vendor impersonation that a human would miss
  2. a duplicate payment that a keyword rule would also catch -- but alongside
     dozens of false alarms
  3. a LEGITIMATE invoice that looks guilty and is correctly released

The third is the one worth dwelling on. Anyone can build something that flags
suspicious-looking invoices. The difficulty is not flagging the honest ones.

    python src/demo.py            narrated by Claude (needs ANTHROPIC_API_KEY)
    python src/demo.py --no-llm   evidence only, no network calls
"""

import sys

import pandas as pd

from agent import RELEASE_BELOW, ESCALATE_ABOVE, build_case, explain, load_agent

WIDTH = 78


def pick_cases(test, model, ledger, changes):
    """
    Choose one invoice of each kind from the sealed test set, by outcome rather than
    by hand-picked ID, so the demo keeps working if the data is regenerated.
    """
    scored = []
    for row in test.itertuples():
        record = test[test.invoice_id == row.invoice_id].iloc[0]
        case = build_case(record, record, model, ledger, changes)
        scored.append((case, record))

    def first(predicate):
        return next((c for c in scored if predicate(*c)), None)

    picks = []
    swap = first(lambda c, r: r.fraud_type == "bank_swap" and c["action"] == "ESCALATE")
    dupe = first(lambda c, r: r.fraud_type == "duplicate_payment" and c["action"] != "RELEASE")
    # An honest invoice that looks guilty: nearly identical to one the company
    # ALREADY PAID, days earlier, for nearly the same amount -- and correctly
    # released anyway. The recency is what makes it hard; a lookalike from two years
    # ago fools nobody. This is the case the keyword rule gets wrong, and the one
    # worth spending demo time on.
    clean = first(
        lambda c, r: r.true_fraud == 0
        and c["action"] == "RELEASE"
        and r.max_similarity_to_prior > 0.93
        and r.amount_ratio_to_closest_match > 0.95
        and r.days_since_closest_match < 45
    )
    for label, found in [
        ("VENDOR IMPERSONATION", swap),
        ("DUPLICATE PAYMENT", dupe),
        ("LOOKS GUILTY, ISN'T", clean),
    ]:
        if found:
            picks.append((label, found[0], found[1]))
    return picks


def show(label, case, record, narrate):
    verdict = {"ESCALATE": "!!", "HOLD": " ?", "RELEASE": " ok"}[case["action"]]

    print()
    print("=" * WIDTH)
    print(f"{label}")
    print("=" * WIDTH)
    print(f"  invoice     {case['invoice_id']}   vendor {case['vendor_id']}")
    print(f"  amount      ${case['amount']:,.2f}")
    print(f"  risk score  {case['risk_score']:.2f}")
    print(f"  decision    {verdict}  {case['action']} - {case['instruction']}")

    print()
    print("  the model's reasoning, in full:")
    for i, step in enumerate(case["decision_path"], 1):
        print(f"    {i}. {step['label']}")
        print(f"       measured {step['value']:,.2f}  |  rule: {step['test']}")

    evidence = case["evidence"]
    if evidence.get("closest_prior_invoice"):
        m = evidence["closest_prior_invoice"]
        credited = "yes - this is a rebill" if m["was_credited_back"] else "NO - still paid"
        print()
        print(f"  closest prior invoice   {m['invoice_id']}")
        plural = "day" if m["days_before"] == 1 else "days"
        print(f"    ${m['amount']:,.2f} on day {m['day']}, {m['days_before']} {plural} before this one")
        print(f"    text match {m['similarity']:.0%}, amount differs by ${abs(m['amount_difference']):,.2f}")
        print(f"    credited back?  {credited}")

    if evidence.get("bank_detail_change"):
        b = evidence["bank_detail_change"]
        print()
        bplural = "day" if b["days_before_this_invoice"] == 1 else "days"
        print(f"  bank details changed    day {b['changed_on_day']}, "
              f"{b['days_before_this_invoice']} {bplural} before this invoice")
        print(f"    {b['old_account']} -> {b['new_account']}")

    if evidence.get("bank_account_shared_with"):
        shared = evidence["bank_account_shared_with"]
        print()
        print(f"  bank account also used by {len(shared)} other vendor record(s): {', '.join(shared)}")

    profile = evidence["vendor_profile"]
    if profile["median_prior_amount"]:
        multiple = profile["this_amount"] / profile["median_prior_amount"]
        print()
        print(f"  vendor history          {profile['prior_invoices']} prior invoices, "
              f"median ${profile['median_prior_amount']:,.2f}")
        print(f"    this invoice is {multiple:.1f}x the vendor's usual")
    print(f"  purchase order          {profile['purchase_order']}")

    if narrate:
        print()
        print("  " + "-" * (WIDTH - 4))
        print("  CASE NOTE FOR THE ANALYST")
        print("  " + "-" * (WIDTH - 4))
        for line in _wrap(case["memo"], WIDTH - 4):
            print(f"  {line}")

    print()
    truth = "FRAUD" if record.true_fraud else "legitimate"
    correct = (record.true_fraud == 1) == (case["action"] != "RELEASE")
    print(f"  ground truth: {truth}   ->   agent was {'RIGHT' if correct else 'WRONG'}")


def _wrap(text, width):
    lines, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main():
    narrate = "--no-llm" not in sys.argv

    model, ledger, changes = load_agent()
    test = pd.read_csv("data/test.csv").fillna({"po_number": "", "fraud_type": ""})

    print("=" * WIDTH)
    print("ACCOUNTS PAYABLE -- PAYMENT RUN REVIEW")
    print("=" * WIDTH)
    print(f"  {len(test)} invoices are queued for payment.")
    print(f"  Risk thresholds: release below {RELEASE_BELOW}, escalate at {ESCALATE_ABOVE} and up.")
    if not narrate:
        print("  (running without Claude narration)")

    picks = pick_cases(test, model, ledger, changes)
    for label, case, record in picks:
        if narrate:
            case["memo"] = explain(case)
        show(label, case, record, narrate)

    print()
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
