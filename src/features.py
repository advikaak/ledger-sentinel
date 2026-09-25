"""
Turn a raw invoice into the numbers a model can learn from.

Two rules govern this file:

1. POINT-IN-TIME. Every feature for invoice X is computed using only invoices that
   arrived BEFORE X. If we let the model see the future it would score beautifully
   in testing and fail the moment it went live.

2. BEHAVIOR, NOT IDENTITY. Every feature below describes something the invoice or
   the vendor DID. None of them describe what the vendor IS. Vendor region, vendor
   name and anything derived from them are deliberately absent -- see EXCLUDED at
   the bottom of this file for why.

Because training and scoring both call compute_features(), the model cannot be fed
one set of numbers in the lab and a different set in production.
"""

from difflib import SequenceMatcher

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "amount_log",
    "amount_vs_vendor_median",
    "is_round_amount",
    "vendor_age_days",
    "prior_invoice_count",
    "days_since_bank_change",
    "max_similarity_to_prior",
    "amount_ratio_to_closest_match",
    "days_since_closest_match",
    "closest_match_was_credited",
    "gap_vs_vendor_cadence",
    "has_purchase_order",
    "accounts_sharing_bank",
    "off_hours_submission",
]

NEVER_CHANGED = 9999.0
NO_MATCH_DAYS = 9999.0

# Human-readable labels used in the case files the agent writes.
FEATURE_LABELS = {
    "amount_log": "invoice amount",
    "amount_vs_vendor_median": "amount relative to this vendor's usual invoice",
    "is_round_amount": "suspiciously round amount",
    "vendor_age_days": "how long this vendor has existed in our system",
    "prior_invoice_count": "number of prior invoices from this vendor",
    "days_since_bank_change": "days since this vendor's bank details changed",
    "max_similarity_to_prior": "text similarity to a previous invoice",
    "amount_ratio_to_closest_match": "amount match against that previous invoice",
    "days_since_closest_match": "days between this invoice and its closest match",
    "closest_match_was_credited": "the invoice it resembles was credited back, making this a legitimate rebill",
    "gap_vs_vendor_cadence": "how early this invoice arrived against the vendor's own billing rhythm",
    "has_purchase_order": "backed by a purchase order",
    "accounts_sharing_bank": "other vendors sharing this bank account",
    "off_hours_submission": "submitted outside business hours",
}


# Short names for charts and the web app's rule chains. One vocabulary everywhere.
FEATURE_SHORT = {
    "amount_log": "Invoice amount",
    "amount_vs_vendor_median": "Amount (× vendor's usual)",
    "is_round_amount": "Round-number amount",
    "vendor_age_days": "Vendor age",
    "prior_invoice_count": "Prior invoices from vendor",
    "days_since_bank_change": "Days since bank-detail change",
    "max_similarity_to_prior": "Text match to closest earlier invoice",
    "amount_ratio_to_closest_match": "Amount match to closest earlier invoice",
    "days_since_closest_match": "Days since closest earlier invoice",
    "closest_match_was_credited": "Closest earlier invoice credited back",
    "gap_vs_vendor_cadence": "Gap since vendor's last invoice (× usual)",
    "has_purchase_order": "Has purchase order",
    "accounts_sharing_bank": "Other vendors on same bank account",
    "off_hours_submission": "Submitted outside business hours",
}


def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _normalize(text, invoice_id):
    """Compare the billing content, and strip the invoice number's suffix so that
    INV-4471-A does not look unrelated to INV-4471."""
    base = invoice_id.split("-")[:2]
    return f"{'-'.join(base)} {text}".lower()


def compute_features(invoice, prior_invoices, bank_changes, bank_account_owners):
    """
    invoice              dict-like, the invoice being scored
    prior_invoices       list of dicts, this vendor's earlier invoices only
    bank_changes         list of dicts for this vendor, each with change_day
    bank_account_owners  {bank_account: set(vendor_ids)} built from prior data
    """
    amount = float(invoice["amount"])
    day = int(invoice["day"])

    prior_amounts = [float(p["amount"]) for p in prior_invoices]
    vendor_median = float(np.median(prior_amounts)) if prior_amounts else amount

    # How far after the most recent bank change did this invoice land?
    past_changes = [c for c in bank_changes if int(c["change_day"]) <= day]
    days_since_change = (
        float(day - max(int(c["change_day"]) for c in past_changes))
        if past_changes
        else NEVER_CHANGED
    )

    # Find the prior invoice this one most resembles textually, then ask how
    # closely the amounts and the timing line up. A recurring monthly bill scores
    # high on similarity but its amount drifts and it arrives a cycle later;
    # a resubmitted duplicate matches on all three at once.
    this_text = _normalize(invoice["description"], invoice["invoice_id"])
    best_similarity, best_ratio, best_gap, best_credited = 0.0, 0.0, NO_MATCH_DAYS, 0.0
    for p in prior_invoices:
        s = _similarity(this_text, _normalize(p["description"], p["invoice_id"]))
        if s > best_similarity:
            best_similarity = s
            p_amount = float(p["amount"])
            best_ratio = min(amount, p_amount) / max(amount, p_amount) if p_amount else 0.0
            best_gap = float(day - int(p["day"]))
            # If the invoice this one resembles was credited back, this is a rebill
            # of a cancelled charge, not a second claim on a charge already paid.
            best_credited = float(int(p.get("cancelled", 0)))

    # How early is this invoice against the vendor's OWN rhythm? A supplier that has
    # billed every thirty days for two years, suddenly billing again after six, is
    # anomalous in a way that "six days" on its own is not -- a weekly vendor billing
    # after six days is simply on time. Absolute gaps cannot tell those apart, which
    # is why a raw duplicate-detection rule drowns in false positives.
    if len(prior_invoices) >= 3:
        days = sorted(int(p["day"]) for p in prior_invoices)
        gaps = [b - a for a, b in zip(days, days[1:]) if b > a]
        typical_gap = float(np.median(gaps)) if gaps else 0.0
        last_gap = float(day - days[-1])
        gap_vs_cadence = last_gap / typical_gap if typical_gap > 0 else 1.0
    else:
        # Too little history to know the vendor's rhythm. Claim nothing.
        gap_vs_cadence = 1.0

    sharing = bank_account_owners.get(invoice["bank_account"], set())
    others_sharing = len(sharing - {invoice["vendor_id"]})

    return {
        "amount_log": float(np.log1p(amount)),
        "amount_vs_vendor_median": amount / vendor_median if vendor_median else 1.0,
        "is_round_amount": float(amount == round(amount, -2)),
        "vendor_age_days": float(max(day - int(invoice["vendor_created_day"]), 0)),
        "prior_invoice_count": float(len(prior_invoices)),
        "days_since_bank_change": days_since_change,
        "max_similarity_to_prior": best_similarity,
        "amount_ratio_to_closest_match": best_ratio,
        "days_since_closest_match": best_gap,
        "closest_match_was_credited": best_credited,
        "gap_vs_vendor_cadence": gap_vs_cadence,
        "has_purchase_order": float(bool(str(invoice.get("po_number") or "").strip())),
        "accounts_sharing_bank": float(others_sharing),
        "off_hours_submission": float(int(invoice["hour"]) < 8 or int(invoice["hour"]) > 18),
    }


def build_feature_table(invoices_df, vendors_df, changes_df):
    """
    Walk the ledger in chronological order, computing each invoice's features from
    what was known at that moment. Slower than a vectorized pass, and correct,
    which matters more.
    """
    vendor_created = vendors_df.set_index("vendor_id")["created_day"].to_dict()
    changes_by_vendor = {}
    for c in changes_df.to_dict("records"):
        changes_by_vendor.setdefault(c["vendor_id"], []).append(c)

    history = {}
    bank_account_owners = {}
    rows = []

    for inv in invoices_df.sort_values(["day", "invoice_id"]).to_dict("records"):
        inv = dict(inv, vendor_created_day=vendor_created[inv["vendor_id"]])
        prior = history.get(inv["vendor_id"], [])
        feats = compute_features(
            inv,
            prior,
            changes_by_vendor.get(inv["vendor_id"], []),
            bank_account_owners,
        )
        rows.append({"invoice_id": inv["invoice_id"], **feats})

        history.setdefault(inv["vendor_id"], []).append(inv)
        bank_account_owners.setdefault(inv["bank_account"], set()).add(inv["vendor_id"])

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# EXCLUDED BY DESIGN -- this list is part of the deliverable, not an omission.
#
#   vendor_region          A vendor's location is not a behavior. Training on it
#                          teaches the model to suspect geography, and because our
#                          historical labels carry past investigators' bias, the
#                          model would learn that bias and call it a pattern.
#   vendor_name features   Name shape is a proxy for national origin. Same problem,
#                          harder to see.
#   submitter identity     Flags the employee, not the transaction.
#
# The naive baseline in train.py deliberately includes vendor_region so the
# evaluation can show what that choice costs.
# --------------------------------------------------------------------------
EXCLUDED_FEATURES = {
    "vendor_region": "location is an attribute, not a behavior; historical labels are biased by region",
    "vendor_name_tokens": "proxy for national origin",
    "submitter_id": "flags the employee rather than the transaction",
}
