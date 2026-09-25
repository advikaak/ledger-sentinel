"""
Ledger Sentinel: web app.

Screens a queue of supplier invoices for fraud. For any invoice it shows the
decision, the exact rule chain the model followed, the records behind it, and an
analyst case note written by Claude.

On first load the app builds everything from source: it generates the synthetic
ledger, trains and evaluates the models, and keeps the result for as long as the
server runs. Nothing is read from a pre-trained file, so the app always matches
the code in this repository.

    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import anthropic
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import analysis as A  # noqa: E402
import demo  # noqa: E402
import evaluate  # noqa: E402
import generate_data  # noqa: E402
import train  # noqa: E402
from agent import ESCALATE_ABOVE, RELEASE_BELOW, build_case, explain, load_agent, route  # noqa: E402
from features import EXCLUDED_FEATURES, FEATURE_LABELS, FEATURE_NAMES, FEATURE_SHORT, NEVER_CHANGED  # noqa: E402

st.set_page_config(page_title="Ledger Sentinel", page_icon=":material/receipt_long:", layout="wide")

DECISIONS = ["ESCALATE", "HOLD", "RELEASE"]
SCHEMES = {
    "bank_swap": "vendor impersonation (bank-detail swap)",
    "duplicate_payment": "duplicate payment",
    "phantom_vendor": "phantom vendor",
}
SHOWCASE = {
    "VENDOR IMPERSONATION": "Impersonation caught",
    "DUPLICATE PAYMENT": "Duplicate caught",
    "LOOKS GUILTY, ISN'T": "Looks guilty, isn't",
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

# Streamlit secrets are loaded into the environment when the server starts.
CLAUDE_CONFIGURED = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ------------------------------------------------------------------ data

@st.cache_resource(show_spinner="Generating the ledger and training the models. This runs once, when the app starts…")
def build():
    for folder in ("data", "models", "out"):
        (ROOT / folder).mkdir(exist_ok=True)
    generate_data.main()
    train.main()
    evaluate.main()

    model, ledger, changes = load_agent()
    bundle, _, test = A.load()
    y = test["true_fraud"].to_numpy()
    probs = A.scored_models(bundle, test)

    queue = test[["invoice_id", "vendor_id", "amount", "true_fraud"]].copy()
    queue["risk"] = probs[A.TREE]
    queue["decision"] = [route(p)[0] for p in queue["risk"]]
    queue["truth"] = np.where(queue["true_fraud"] == 1, "Fraud", "Legitimate")
    queue = queue.sort_values("risk", ascending=False, kind="stable").reset_index(drop=True)

    showcase = {label: case["invoice_id"] for label, case, _ in demo.pick_cases(test, model, ledger, changes)}

    return {
        "model": model,
        "ledger": ledger,
        "changes": changes,
        "test": test,
        "queue": queue,
        "showcase": showcase,
        "metrics": A.metrics_table(y, probs, A.keyword_rule(test)),
        "fairness": A.fairness_table(test, A.hard(probs[A.TREE]), A.hard(probs[A.NAIVE])),
        "routing": A.routing_table(test, probs[A.TREE]),
    }


@st.cache_data(show_spinner=False, max_entries=2000)
def case_for(invoice_id):
    art = build()
    record = art["test"][art["test"]["invoice_id"] == invoice_id].iloc[0]
    return build_case(record, record, art["model"], art["ledger"], art["changes"])


@st.cache_data(show_spinner=False, max_entries=2000)
def case_note(invoice_id):
    """One Claude call per invoice for the life of the server, however many
    visitors ask. That caps what a public deployment can spend."""
    return explain(case_for(invoice_id))


# ------------------------------------------------------------------ formatting

def md_money(value):
    """Dollar amount for markdown. The backslash stops Streamlit from reading
    a pair of dollar signs as a maths formula."""
    return f"\\${value:,.2f}"


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
        return md_money(float(np.expm1(value)))
    raise KeyError(feature)


def step_markdown(i, step):
    feature, value, threshold = step["feature"], step["value"], step["threshold"]
    line = f"**{i}. {FEATURE_SHORT[feature]}:** {feature_value(feature, value)}"
    if feature not in BINARY:
        side = "at most" if value <= threshold else "more than"
        line += f" (rule: {side} {feature_value(feature, threshold)})"
    return line


# ------------------------------------------------------------------ interaction

def open_invoice(invoice_id):
    st.session_state.invoice_id = invoice_id
    # A new table key re-draws the table with the chosen invoice highlighted.
    st.session_state.table_version += 1


def open_from_table(table_key, visible_ids):
    rows = st.session_state[table_key]["selection"]["rows"]
    if rows:
        st.session_state.invoice_id = visible_ids[rows[0]]


# ------------------------------------------------------------------ page

art = build()
queue = art["queue"]
st.session_state.setdefault("invoice_id", art["showcase"]["VENDOR IMPERSONATION"])
st.session_state.setdefault("table_version", 0)
st.session_state.setdefault("notes", {})

st.title("Ledger Sentinel")
st.markdown(
    "An AI agent that screens supplier invoices for fraud before they're paid, "
    "and explains every decision it makes."
)
st.caption(
    "All data is synthetic: a simulated two-year ledger with planted duplicate payments, vendor "
    "impersonations and phantom vendors, so every decision can be checked against ground truth. "
    f"The queue below is the held-out test period: the last {len(queue):,} invoices, which the models never saw."
)

routing = art["routing"]
reviewed = routing.loc[["ESCALATE", "HOLD"], "invoices"].sum() / routing["invoices"].sum()
caught = routing.loc[["ESCALATE", "HOLD"], "fraudulent"].sum() / routing["fraudulent"].sum()
dollars = routing.loc[["ESCALATE", "HOLD"], "fraud_dollars"].sum() / routing["fraud_dollars"].sum()
baseline = art["metrics"].loc[A.RULE, "recall"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fraud caught", f"{caught:.0%}", help="Share of fraudulent invoices the agent escalated or held.", border=True)
k2.metric("Invoices sent to review", f"{reviewed:.0%}", help="Share of the queue an analyst has to look at.", border=True)
k3.metric("Fraudulent dollars intercepted", f"{dollars:.0%}", help="Share of fraudulent dollars in the escalated or held invoices.", border=True)
k4.metric("Keyword rule catches", f"{baseline:.0%}", help="The baseline: flag anything nearly identical to an earlier invoice.", border=True)

tab_queue, tab_perf, tab_fair, tab_how = st.tabs(["Payment queue", "Performance", "Fairness audit", "How it works"])

# ------------------------------------------------------------------ tab: queue

with tab_queue:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("**Start with one of these, or pick any invoice in the queue.**")
        pick_cols = st.columns(len(SHOWCASE))
        for col, (label, button_text) in zip(pick_cols, SHOWCASE.items()):
            col.button(
                button_text,
                on_click=open_invoice,
                args=(art["showcase"][label],),
                width="stretch",
                type="primary" if st.session_state.invoice_id == art["showcase"][label] else "secondary",
            )

        shown = st.pills("Decision", DECISIONS, selection_mode="multi", default=DECISIONS, key="decision_filter")
        view = queue[queue["decision"].isin(shown)].reset_index(drop=True)
        visible_ids = view["invoice_id"].tolist()
        table_key = f"queue_{st.session_state.table_version}_{'-'.join(sorted(shown))}"
        current = st.session_state.invoice_id
        default_rows = [visible_ids.index(current)] if current in visible_ids else []

        st.dataframe(
            view[["invoice_id", "vendor_id", "amount", "risk", "decision", "truth"]],
            key=table_key,
            on_select=lambda: open_from_table(table_key, visible_ids),
            selection_mode="single-row",
            selection_default={"selection": {"rows": default_rows}},
            hide_index=True,
            height=460,
            width="stretch",
            column_config={
                "invoice_id": st.column_config.TextColumn("Invoice"),
                "vendor_id": st.column_config.TextColumn("Vendor"),
                "amount": st.column_config.NumberColumn("Amount", format="dollar"),
                "risk": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
                "decision": st.column_config.TextColumn("Decision"),
                "truth": st.column_config.TextColumn("Ground truth"),
            },
        )
        st.caption(
            f"{plural(len(view), 'invoice')} shown, highest risk first. Escalate at {ESCALATE_ABOVE}, "
            f"hold for review from {RELEASE_BELOW}, release below. Select a row to open it."
        )

    with right:
        case = case_for(current)
        record = art["test"][art["test"]["invoice_id"] == current].iloc[0]

        with st.container(border=True):
            st.subheader(f"{case['invoice_id']} · {case['vendor_id']} · {md_money(case['amount'])}")
            decision_box = {"ESCALATE": (st.error, ":material/block:"),
                            "HOLD": (st.warning, ":material/pause_circle:"),
                            "RELEASE": (st.success, ":material/check_circle:")}
            show_box, icon = decision_box[case["action"]]
            show_box(f"**{case['action'].title()}:** {case['instruction']}.", icon=icon)
            st.markdown(f"Risk score **{case['risk_score']:.2f}**")

            st.markdown("##### The model's reasoning, in full")
            for i, step in enumerate(case["decision_path"], 1):
                st.markdown(step_markdown(i, step))
                st.caption(f"`{step['test']}`")

            st.markdown("##### The records behind it")
            evidence = case["evidence"]
            match = evidence.get("closest_prior_invoice")
            if match:
                credited = "yes, so this is a corrected rebill" if match["was_credited_back"] else "no, it was paid and stayed paid"
                st.markdown(
                    f"- **Closest earlier invoice:** {match['invoice_id']}, {md_money(match['amount'])}, "
                    f"{plural(match['days_before'], 'day')} earlier. Text match {match['similarity']:.0%}; "
                    f"amounts differ by {md_money(abs(match['amount_difference']))}. Credited back: {credited}."
                )
            change = evidence.get("bank_detail_change")
            if change:
                st.markdown(
                    f"- **Bank details changed** {plural(change['days_before_this_invoice'], 'day')} before "
                    f"this invoice: {change['old_account']} → {change['new_account']}."
                )
            shared = evidence.get("bank_account_shared_with")
            if shared:
                st.markdown(f"- **Bank account also used by** {', '.join(shared)}.")
            profile = evidence["vendor_profile"]
            if profile["median_prior_amount"]:
                st.markdown(
                    f"- **Vendor history:** {plural(profile['prior_invoices'], 'earlier invoice')}, usually "
                    f"{md_money(profile['median_prior_amount'])}; this one is "
                    f"{profile['this_amount'] / profile['median_prior_amount']:.1f}× that."
                )
            else:
                st.markdown("- **Vendor history:** no earlier invoices.")
            st.markdown(f"- **Purchase order:** {profile['purchase_order'].replace('NONE', 'none')}")

            truth = "fraud, " + SCHEMES[record["fraud_type"]] if record["true_fraud"] else "legitimate"
            correct = bool(record["true_fraud"]) == (case["action"] != "RELEASE")
            st.markdown(
                f"**Ground truth (synthetic):** {truth}. "
                + ("The agent's decision was **correct**." if correct else "The agent's decision was **wrong**.")
            )

            st.markdown("##### Case note for the analyst")
            if CLAUDE_CONFIGURED:
                if st.button("Write case note with Claude", icon=":material/edit_note:", key=f"note_{current}"):
                    with st.spinner("Claude is writing the note…"):
                        try:
                            st.session_state.notes[current] = case_note(current)
                        except anthropic.APIError as error:
                            st.error(f"The Claude request failed: {error}")
                if current in st.session_state.notes:
                    st.info(st.session_state.notes[current].replace("$", "\\$"), icon=":material/description:")
                st.caption("Claude writes up the model's decision and the records above. It is never asked whether the invoice is fraud.")
            else:
                st.caption("Case notes are written by Claude and need an API key, which isn't configured on this deployment.")

# ------------------------------------------------------------------ tab: performance

with tab_perf:
    st.markdown(
        "Four models trained on identical features and identical labels, tuned on a validation period, "
        "then scored once on the sealed test period. The keyword rule is the baseline: it flags any invoice "
        "nearly identical to an earlier one."
    )
    metrics = art["metrics"]
    st.dataframe(
        {
            "Model": metrics.index.tolist(),
            "Precision": [f"{v:.0%}" for v in metrics["precision"]],
            "Recall": [f"{v:.0%}" for v in metrics["recall"]],
            "F1": [f"{v:.2f}" for v in metrics["f1"]],
            "Average precision": ["—" if np.isnan(v) else f"{v:.2f}" for v in metrics["average_precision"].astype(float)],
            "Flagged": metrics["flagged"].astype(int).tolist(),
            "Missed": metrics["missed"].astype(int).tolist(),
        },
        hide_index=True,
        width="stretch",
    )
    st.image("out/figures/02_precision_recall.png", width=780)
    st.markdown(
        "**Why ship the tree when the forest scores higher?** Holding a supplier's payment has to be "
        "justified to that supplier and to an auditor. The tree gives at most six readable conditions for "
        "every decision; the forest averages 300 trees and can't show its work. A paired bootstrap confirms "
        "the forest's lead is real, not noise, so this is a deliberate trade-off."
    )
    st.image("out/figures/03_uncertainty.png", width=780)

# ------------------------------------------------------------------ tab: fairness

with tab_fair:
    st.markdown(
        "The models train on what past investigators **wrote down**, not on ground truth. In this ledger one "
        "region spent years under an enhanced-scrutiny policy, so about a fifth of its clean invoices carry a "
        "fraud label they never earned. Two otherwise identical models were trained on those labels; one was "
        "also allowed to see the vendor's region. The table shows how often each flags **truly clean** invoices."
    )
    fairness = art["fairness"]
    rows = []
    for region in fairness.sort_values("rate", ascending=False)["region"].unique():
        part = fairness[fairness["region"] == region].set_index("model")
        row = {"Region": region, "Clean invoices": int(part["clean_invoices"].iloc[0])}
        for model, column in ((A.TREE, "Shipped model"), (A.NAIVE, "Model that sees region")):
            r = part.loc[model]
            row[column] = f"{r['rate']:.1%} ({r['ci_low']:.1%}–{r['ci_high']:.1%})"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("False-alarm rate on clean invoices, with 95% Wilson intervals.")
    st.image("out/figures/04_fairness.png", width=780)
    st.markdown(
        "1. **The region-aware model learned the bias and treated it as a pattern.** It flags about three in "
        "five clean South Asian invoices, and its overall precision is lower.\n"
        "2. **Removing the column reduced the disparity but did not remove it.** The biased labels are still "
        "in the training data, and the model reaches some of the same conclusions through behavioural "
        "features. Dropping a sensitive column is a first step, not a fix; the labels themselves would need "
        "to be re-adjudicated."
    )

# ------------------------------------------------------------------ tab: how it works

with tab_how:
    st.markdown(
        "**The agent, for every invoice in the payment queue:**\n"
        "1. **Perceive**: pull the invoice and the vendor's history.\n"
        "2. **Decide**: score it with an interpretable decision tree and recover the exact rule chain it followed.\n"
        "3. **Act**: release it, hold it for review, or escalate and block payment.\n"
        "4. **Explain**: hand the rule chain and the records to Claude, which writes the analyst's case note."
    )
    st.markdown(
        "**Why the obvious rule fails.** Most honest invoices look nearly identical to an earlier one, because "
        "suppliers send the same bill every cycle. The signal is timing: a duplicate arrives too early for that "
        "vendor's usual billing rhythm."
    )
    st.image("out/figures/01_similarity_vs_timing.png", width=920)

    st.markdown("**Features the model uses.** Each one describes something the invoice or vendor *did*, "
                "computed only from records that existed before the invoice arrived.")
    st.dataframe(
        {
            "Feature": [FEATURE_SHORT[f] for f in FEATURE_NAMES],
            "What it measures": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
            "Column": FEATURE_NAMES,
        },
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Features deliberately excluded.**")
    st.dataframe(
        {"Excluded": list(EXCLUDED_FEATURES), "Reason": list(EXCLUDED_FEATURES.values())},
        hide_index=True,
        width="stretch",
    )
    st.image("out/figures/05_permutation_importance.png", width=780)
    st.markdown(
        "**Method.** Time-based train, validation and test split; hyperparameters tuned on validation with "
        "an explicit cost ratio (a missed fraud costs six false alarms); precision–recall rather than ROC "
        "because fraud is rare; paired-bootstrap intervals; permutation importance on held-out data.\n\n"
        "**Limitations.** The data is synthetic, the fraud rate (about 5%) is far higher than real life, the "
        "test period holds only 31 fraud cases, and no adversary adapts to the model. The results show the "
        "method works on planted patterns, not what accuracy would be on a real ledger."
    )
