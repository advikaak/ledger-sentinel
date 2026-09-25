"""
Synthesize a vendor-payment ledger with planted fraud.

Why synthetic: real accounts-payable ledgers are confidential and labeled fraud
cases are rarer still. We generate a world where we know the ground truth, so we
can measure whether the model finds the patterns we planted.

The hard part is making the LEGITIMATE invoices look superficially suspicious --
recurring vendors bill near-identical invoices every month, real vendors really do
change bank accounts, and plenty of honest invoices are round numbers. If only
fraud had those traits the model would learn one giveaway feature and the whole
project would be hollow. Here the signal is in the COMBINATION and the TIMING.
"""

import random
from dataclasses import dataclass, asdict

import pandas as pd

SEED = 20261113
N_VENDORS = 140
HORIZON_DAYS = 730

REGIONS = ["North America", "Europe", "South Asia", "East Asia", "Latin America"]

# One region spent years under a standing "enhanced scrutiny" policy, so a share of
# its perfectly clean invoices got written up as suspicious. Those write-ups are the
# only labels a real team would have to train on.
#
# This rate is a DIAL WE SET, not a finding -- say so out loud when presenting. At
# 0.20, one in five clean invoices from that region carries a fraud label it never
# earned. The effect below holds anywhere from roughly 0.15 to 0.28; we report one
# value rather than the sweep for legibility.
OVER_SCRUTINIZED_REGION = "South Asia"
HISTORICAL_FALSE_FLAG_RATE = 0.20

SERVICES = [
    "Managed logistics services",
    "Freight forwarding",
    "Warehouse labor contract",
    "Customs brokerage fees",
    "Packaging materials",
    "Cold chain monitoring",
    "Fleet maintenance",
    "Pallet supply",
    "Last mile delivery",
    "Inventory audit services",
    "Component assembly",
    "Quality inspection services",
]


@dataclass
class Vendor:
    vendor_id: str
    name: str
    region: str
    created_day: int
    bank_account: str
    cadence_days: int
    base_amount: float
    service: str


@dataclass
class Invoice:
    invoice_id: str
    vendor_id: str
    day: int
    hour: int
    amount: float
    description: str
    po_number: str
    bank_account: str
    # Was this invoice cancelled / credited back after it was raised? A legitimate
    # rebill is preceded by a credit note against the original. A fraudulent
    # duplicate is not -- the original was paid and stayed paid. This is the control
    # an AP department actually relies on, and it is the only thing that reliably
    # separates the two once text and amount have failed.
    cancelled: int
    true_fraud: int
    fraud_type: str


def _account(rng):
    return f"{rng.randint(10**9, 10**10 - 1)}"


def build_vendors(rng):
    """Most vendors are long-standing. A minority are new, which is normal."""
    vendors = []
    for i in range(N_VENDORS):
        # Long tail of vendor age. Some genuinely new vendors exist and are fine.
        created = rng.choice(
            [rng.randint(-1200, -400), rng.randint(-400, -60), rng.randint(-60, 500)],
        )
        vendors.append(
            Vendor(
                vendor_id=f"V{1000 + i}",
                name=f"Vendor {1000 + i}",
                region=rng.choice(REGIONS),
                created_day=created,
                bank_account=_account(rng),
                cadence_days=rng.choice([30, 30, 30, 14, 60, 90]),
                base_amount=round(rng.uniform(1800, 48000), 2),
                service=rng.choice(SERVICES),
            )
        )
    return vendors


def legitimate_invoices(vendors, rng):
    """
    Normal billing. Recurring vendors produce near-identical descriptions month
    after month with amounts that drift a few percent -- this is what makes naive
    text-similarity dedupe useless in the real world.
    """
    invoices = []
    bank_changes = []
    counter = 40000

    # Real companies have vendor records that legitimately share a bank account:
    # subsidiaries of one parent billing under separate entities, or a factoring
    # company collecting for several suppliers. If only fraud shared accounts, the
    # feature would be a giveaway and the model would have nothing to learn.
    for i in range(0, len(vendors) - 1, 12):
        vendors[i + 1].bank_account = vendors[i].bank_account

    for v in vendors:
        # ~7% of honest vendors change bank details during the window. Real thing:
        # acquisitions, bank consolidation, switching to a new treasury provider.
        if rng.random() < 0.07 and v.created_day < HORIZON_DAYS - 120:
            change_day = rng.randint(max(v.created_day + 30, 10), HORIZON_DAYS - 60)
            new_account = _account(rng)
            bank_changes.append(
                {
                    "vendor_id": v.vendor_id,
                    "change_day": change_day,
                    "old_account": v.bank_account,
                    "new_account": new_account,
                    "legitimate": 1,
                }
            )
        else:
            change_day, new_account = None, None

        day = max(v.created_day, 0) + rng.randint(0, v.cadence_days)
        while day < HORIZON_DAYS:
            counter += 1
            amount = round(v.base_amount * rng.uniform(0.88, 1.14), 2)
            # Honest invoices spike too: annual renewals, bulk orders, year-end
            # true-ups. Without these, "much larger than usual" would be a free
            # giveaway for the bank-swap scheme and the model would look brilliant
            # on data that flattered it.
            if rng.random() < 0.05:
                amount = round(amount * rng.uniform(2.0, 4.2), 2)
            # Honest round-number invoices: fixed-fee retainers and flat contracts.
            if rng.random() < 0.12:
                amount = float(round(amount, -2))
            account = (
                new_account if change_day is not None and day > change_day else v.bank_account
            )
            invoices.append(
                Invoice(
                    invoice_id=f"INV-{counter}",
                    vendor_id=v.vendor_id,
                    day=day,
                    # Honest invoices land overnight more often than people expect:
                    # overseas vendors in other time zones, and automated batch
                    # uploads that run after close. Roughly 1 in 8.
                    hour=(
                        rng.choice([9, 10, 11, 13, 14, 15, 16, 17])
                        if rng.random() < 0.87
                        else rng.choice([2, 5, 6, 7, 20, 21, 22, 23])
                    ),
                    amount=amount,
                    description=f"{v.service} - period ending day {day}",
                    # ~80% of honest invoices carry a purchase order. The rest are
                    # emergency buys and low-value spend under the PO threshold.
                    po_number=f"PO-{rng.randint(70000, 79999)}" if rng.random() < 0.80 else "",
                    bank_account=account,
                    cancelled=0,
                    true_fraud=0,
                    fraud_type="",
                )
            )

            # Legitimate reissue: the first invoice had a typo, a wrong cost centre
            # or a disputed line, so it was cancelled and rebilled days later for
            # nearly the same amount. This is the honest transaction that looks
            # exactly like duplicate-payment fraud, and it is why precision is hard.
            if rng.random() < 0.035:
                counter += 1
                # The credit note against the original is what makes this rebill
                # legitimate. About 1 in 6 never gets recorded, because AP teams are
                # busy -- so this is a strong signal, not a perfect one.
                if rng.random() < 0.84:
                    invoices[-1].cancelled = 1
                # Corrected invoices genuinely do get issued as "-R" or "-A" against
                # the original number, which is exactly what a resubmitted duplicate
                # looks like. The numbering carries no information about intent.
                reissue_id = (
                    f"{invoices[-1].invoice_id}-{rng.choice(['R', 'A', 'C'])}"
                    if rng.random() < 0.5
                    else f"INV-{counter}"
                )
                invoices.append(
                    Invoice(
                        invoice_id=reissue_id,
                        vendor_id=v.vendor_id,
                        day=min(day + rng.randint(2, 9), HORIZON_DAYS - 1),
                        hour=rng.choice([9, 11, 14, 16, 17]),
                        amount=round(amount * rng.uniform(0.98, 1.02), 2),
                        description=f"{v.service} - period ending day {day}",
                        po_number=f"PO-{rng.randint(70000, 79999)}" if rng.random() < 0.6 else "",
                        bank_account=account,
                        cancelled=0,
                    true_fraud=0,
                        fraud_type="",
                    )
                )

            day += v.cadence_days + rng.randint(-4, 4)

    return invoices, bank_changes, counter


def inject_duplicate_payments(invoices, vendors, rng, counter, n=55):
    """
    Duplicate payment fraud: an already-paid invoice is resubmitted with a mutated
    invoice number and a tiny amount jitter so exact-match dedupe misses it.

    Distinguishing mark vs. a legitimate recurring invoice: it lands a handful of
    days after the original instead of a full billing cycle later, and the amount
    matches to within ~1% instead of drifting normally.
    """
    by_vendor = {}
    for inv in invoices:
        by_vendor.setdefault(inv.vendor_id, []).append(inv)
    eligible = [v for v, rows in by_vendor.items() if len(rows) >= 4]

    planted = []
    for _ in range(n):
        vid = rng.choice(eligible)
        original = rng.choice(sorted(by_vendor[vid], key=lambda r: r.day)[:-1])
        counter += 1
        # Not every fraudster is careless. A third wait longer and jitter the amount
        # more, which drags them toward the legitimate-reissue pattern on purpose.
        careless = rng.random() < 0.65
        gap = rng.randint(2, 11) if careless else rng.randint(12, 40)
        jitter = rng.uniform(0.995, 1.005) if careless else rng.uniform(0.96, 1.04)
        # Half resubmit under the original number with a suffix, half under a fresh
        # number. If all fraud used suffixes the suffix would BE the label, and the
        # model would learn our generator instead of the fraud.
        if rng.random() < 0.5:
            fraud_id = f"{original.invoice_id}-{rng.choice(['A', 'R', '1', 'B'])}"
        else:
            counter += 1
            fraud_id = f"INV-{counter}"
        planted.append(
            Invoice(
                invoice_id=fraud_id,
                vendor_id=vid,
                day=min(original.day + gap, HORIZON_DAYS - 1),
                hour=rng.choice([9, 11, 14, 16, 20, 22]),
                amount=round(original.amount * jitter, 2),
                description=original.description,
                po_number=original.po_number if rng.random() < 0.45 else "",
                bank_account=original.bank_account,
                cancelled=0,
                true_fraud=1,
                fraud_type="duplicate_payment",
            )
        )
    return planted, counter


def inject_bank_swap(invoices, vendors, rng, counter, bank_changes, n=45):
    """
    Vendor impersonation / business email compromise: an attacker emails AP posing
    as a known vendor, gets the payment details changed, then invoices quickly for
    an unusually large amount before anyone notices.

    Distinguishing mark vs. a legitimate bank change: the invoice follows the change
    within days, and the amount is well above the vendor's normal range.
    """
    by_vendor = {}
    for inv in invoices:
        by_vendor.setdefault(inv.vendor_id, []).append(inv)
    vendor_map = {v.vendor_id: v for v in vendors}
    already_changed = {c["vendor_id"] for c in bank_changes}
    eligible = [
        v for v, rows in by_vendor.items() if len(rows) >= 6 and v not in already_changed
    ]

    planted = []
    for _ in range(n):
        if not eligible:
            break
        vid = eligible.pop(rng.randrange(len(eligible)))
        vendor = vendor_map[vid]
        rows = sorted(by_vendor[vid], key=lambda r: r.day)
        change_day = rng.randint(rows[2].day, max(rows[-1].day - 10, rows[2].day + 1))
        fraud_account = _account(rng)
        bank_changes.append(
            {
                "vendor_id": vid,
                "change_day": change_day,
                "old_account": vendor.bank_account,
                "new_account": fraud_account,
                "legitimate": 0,
            }
        )
        counter += 1
        planted.append(
            Invoice(
                invoice_id=f"INV-{counter}",
                vendor_id=vid,
                day=min(change_day + rng.randint(1, 12), HORIZON_DAYS - 1),
                hour=rng.choice([9, 12, 15, 18, 21, 23]),
                # Attackers usually go big, but the patient ones stay inside the
                # vendor's normal range to avoid tripping an amount threshold.
                amount=round(
                    vendor.base_amount
                    * (rng.uniform(2.1, 4.5) if rng.random() < 0.6 else rng.uniform(0.9, 1.6)),
                    2,
                ),
                # An impersonator who has been reading the vendor's email copies the
                # vendor's own billing language. If fraud used distinctive wording,
                # the model would catch it on text novelty and never learn to look
                # at the bank change -- which is the signal that actually generalizes.
                description=f"{vendor.service} - period ending day {change_day}",
                po_number="" if rng.random() < 0.7 else f"PO-{rng.randint(70000, 79999)}",
                bank_account=fraud_account,
                cancelled=0,
                true_fraud=1,
                fraud_type="bank_swap",
            )
        )
    return planted, counter


def inject_phantom_vendors(rng, counter, n=22):
    """
    Phantom vendor: a fictitious supplier set up to receive payments, often sharing
    a bank account with another vendor record. Bills round numbers, no purchase
    order, no history.
    """
    planted_vendors, planted_invoices = [], []
    shared_pool = [_account(rng) for _ in range(6)]

    for i in range(n):
        # Spread across the whole window. Clustering these at the end would leave
        # the training set with almost no examples of the scheme, and the model
        # would look bad at catching something it had never been shown.
        created = rng.randint(90, HORIZON_DAYS - 20)
        vid = f"V9{100 + i}"
        service = rng.choice(SERVICES)
        # Phantom vendors often cluster on a handful of bank accounts -- the classic
        # tell. The more careful ones open a fresh account per shell, so this cannot
        # be treated as a reliable signature on its own.
        account = rng.choice(shared_pool) if rng.random() < 0.6 else _account(rng)
        planted_vendors.append(
            Vendor(
                vendor_id=vid,
                name=f"Vendor 9{100 + i}",
                region=rng.choice(REGIONS),
                created_day=created,
                bank_account=account,
                cadence_days=30,
                base_amount=0.0,
                service=service,
            )
        )
        for _ in range(rng.randint(1, 3)):
            counter += 1
            planted_invoices.append(
                Invoice(
                    invoice_id=f"INV-{counter}",
                    vendor_id=vid,
                    day=min(created + rng.randint(2, 40), HORIZON_DAYS - 1),
                    hour=rng.choice([7, 19, 20, 21, 22, 23]),
                    amount=float(rng.choice([5000, 7500, 12000, 15000, 22500, 30000])),
                    description=f"{service} - consulting engagement",
                    # Insider-assisted phantom billing comes with a real purchase
                    # order, because someone inside raised it.
                    po_number="" if rng.random() < 0.75 else f"PO-{rng.randint(70000, 79999)}",
                    bank_account=account,
                    cancelled=0,
                true_fraud=1,
                    fraud_type="phantom_vendor",
                )
            )
    return planted_vendors, planted_invoices, counter


def apply_historical_bias(invoices_df, vendors_df, rng):
    """
    The label a real team would actually train on is not ground truth -- it is what
    past investigators WROTE DOWN. If one region was over-scrutinized, some of its
    clean invoices carry a fraud label they never deserved.

    We keep both columns so the evaluation can train on the biased label and score
    against the truth, which is exactly the situation a real compliance team is in.
    """
    region = vendors_df.set_index("vendor_id")["region"]
    historical = invoices_df["true_fraud"].tolist()
    for i, row in enumerate(invoices_df.itertuples()):
        if row.true_fraud == 0 and region.get(row.vendor_id) == OVER_SCRUTINIZED_REGION:
            if rng.random() < HISTORICAL_FALSE_FLAG_RATE:
                historical[i] = 1
    invoices_df["historical_label"] = historical
    return invoices_df


def build_ledger(seed=SEED):
    """Build the whole synthetic world in memory. Same seed, same ledger, every time."""
    rng = random.Random(seed)

    vendors = build_vendors(rng)
    invoices, bank_changes, counter = legitimate_invoices(vendors, rng)

    dupes, counter = inject_duplicate_payments(invoices, vendors, rng, counter)
    swaps, counter = inject_bank_swap(invoices, vendors, rng, counter, bank_changes)
    phantom_vendors, phantom_invoices, counter = inject_phantom_vendors(rng, counter)

    vendors += phantom_vendors
    invoices += dupes + swaps + phantom_invoices

    vendors_df = pd.DataFrame([asdict(v) for v in vendors])
    invoices_df = pd.DataFrame([asdict(i) for i in invoices])
    invoices_df = invoices_df.sort_values(["day", "invoice_id"]).reset_index(drop=True)
    invoices_df = apply_historical_bias(invoices_df, vendors_df, rng)
    changes_df = pd.DataFrame(bank_changes)
    return vendors_df, invoices_df, changes_df


def main():
    vendors_df, invoices_df, changes_df = build_ledger()

    vendors_df.to_csv("data/vendors.csv", index=False)
    invoices_df.to_csv("data/invoices.csv", index=False)
    changes_df.to_csv("data/bank_changes.csv", index=False)

    n = len(invoices_df)
    fraud = int(invoices_df["true_fraud"].sum())
    print(f"vendors              {len(vendors_df)}")
    print(f"invoices             {n}")
    print(f"true fraud           {fraud}  ({fraud / n:.1%})")
    print(f"historical label     {int(invoices_df['historical_label'].sum())} flagged")
    print(f"  of which wrong     {int((invoices_df['historical_label'] > invoices_df['true_fraud']).sum())}")
    print(f"bank changes         {len(changes_df)}  ({int(changes_df['legitimate'].sum())} legitimate)")
    print()
    print(invoices_df[invoices_df.true_fraud == 1]["fraud_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
