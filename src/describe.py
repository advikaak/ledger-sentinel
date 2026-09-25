"""
Turn an agent case into plain English: the rule chain behind a decision and the
records it rests on. The website shows these strings as they are, so all the
wording lives here, in one place, where it can be tested.
"""

import numpy as np

from features import FEATURE_NAMES, FEATURE_SHORT, NEVER_CHANGED

SCHEMES = {
    "bank_swap": "vendor impersonation via a bank-detail swap",
    "duplicate_payment": "duplicate payment",
    "phantom_vendor": "phantom vendor",
}

# How each feature's value reads to a person. Every feature must appear exactly once.
BINARY = {"has_purchase_order", "is_round_amount", "off_hours_submission", "closest_match_was_credited"}
DAYS = {"vendor_age_days", "days_since_bank_change", "days_since_closest_match"}
PERCENT = {"max_similarity_to_prior", "amount_ratio_to_closest_match"}
MULTIPLE = {"gap_vs_vendor_cadence", "amount_vs_vendor_median"}
COUNT = {"prior_invoice_count", "accounts_sharing_bank"}
LOG_DOLLARS = {"amount_log"}
assert sorted(BINARY | DAYS | PERCENT | MULTIPLE | COUNT | LOG_DOLLARS) == sorted(FEATURE_NAMES), (
    "every model feature needs a display format"
)


def money(value):
    return f"${value:,.2f}"


def plural(n, word):
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def feature_value(feature, value):
    if feature in BINARY:
        return "yes" if value >= 0.5 else "no"
    if feature in DAYS:
        return "none on record" if value >= NEVER_CHANGED else plural(int(round(value)), "day")
    if feature in PERCENT:
        return f"{value:.1%}"
    if feature in MULTIPLE:
        return f"{value:.2f}×"
    if feature in COUNT:
        return f"{value:,.0f}"
    if feature in LOG_DOLLARS:
        return money(float(np.expm1(value)))
    raise KeyError(feature)


def describe_step(step):
    """One question the decision tree asked: what it measured and the rule it applied.
    Yes/no features need no rule text; the answer is the rule."""
    feature, value, threshold = step["feature"], step["value"], step["threshold"]
    rule = None
    if feature not in BINARY:
        side = "at most" if value <= threshold else "more than"
        rule = f"{side} {feature_value(feature, threshold)}"
    return {
        "label": FEATURE_SHORT[feature],
        "value": feature_value(feature, value),
        "rule": rule,
        "code": step["test"],
    }


def describe_evidence(evidence):
    lines = []
    match = evidence.get("closest_prior_invoice")
    if match:
        credited = (
            "yes, so this is a corrected rebill" if match["was_credited_back"] else "no, it was paid and stayed paid"
        )
        lines.append({
            "label": "Closest earlier invoice",
            "text": (
                f"{match['invoice_id']}, {money(match['amount'])}, {plural(match['days_before'], 'day')} earlier. "
                f"Text match {match['similarity']:.0%}; amounts differ by {money(abs(match['amount_difference']))}. "
                f"Credited back: {credited}."
            ),
        })
    change = evidence.get("bank_detail_change")
    if change:
        lines.append({
            "label": "Bank details changed",
            "text": (
                f"{plural(change['days_before_this_invoice'], 'day')} before this invoice: "
                f"{change['old_account']} → {change['new_account']}."
            ),
        })
    shared = evidence.get("bank_account_shared_with")
    if shared:
        lines.append({"label": "Bank account also used by", "text": ", ".join(shared) + "."})
    profile = evidence["vendor_profile"]
    if profile["median_prior_amount"]:
        lines.append({
            "label": "Vendor history",
            "text": (
                f"{plural(profile['prior_invoices'], 'earlier invoice')}, usually {money(profile['median_prior_amount'])}; "
                f"this one is {profile['this_amount'] / profile['median_prior_amount']:.1f}× that."
            ),
        })
    else:
        lines.append({"label": "Vendor history", "text": "no earlier invoices."})
    po = profile["purchase_order"]
    lines.append({"label": "Purchase order", "text": "none" if po == "NONE" else po})
    return lines


def describe_case(case, record):
    """Everything the website shows about one invoice."""
    is_fraud = bool(record["true_fraud"])
    return {
        "action": case["action"],
        "instruction": case["instruction"],
        "risk": case["risk_score"],
        "steps": [describe_step(s) for s in case["decision_path"]],
        "evidence": describe_evidence(case["evidence"]),
        "truth": f"fraudulent ({SCHEMES[record['fraud_type']]})" if is_fraud else "legitimate",
        "correct": is_fraud == (case["action"] != "RELEASE"),
    }
