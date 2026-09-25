"""
Runs the Streamlit app headlessly and checks what a visitor would see. The first
run builds the ledger and trains the models, so it takes a little while.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import agent

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 300


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_headline_numbers(app):
    assert app.title[0].value == "Ledger Sentinel"
    values = {m.label: m.value for m in app.metric}
    assert values == {
        "Fraud caught": "90%",
        "Invoices sent to review": "12%",
        "Fraudulent dollars intercepted": "93%",
        "Keyword rule catches": "16%",
    }


def test_opens_on_the_impersonation_case(app):
    assert app.subheader[0].value.startswith("INV-43012")
    assert app.error[0].value.startswith("**Escalate:**")
    assert any("The agent's decision was **correct**" in m.value for m in app.markdown)


def test_showcase_button_opens_the_honest_lookalike(app):
    button(app, "Looks guilty, isn't").click().run()
    assert not app.exception
    assert app.subheader[0].value.startswith("INV-42162")
    assert app.success[0].value.startswith("**Release:**")
    assert any("Credited back: yes" in m.value for m in app.markdown)


def test_case_notes_are_off_without_a_key(app):
    assert not any(b.label == "Write case note with Claude" for b in app.button)
    assert any("isn't configured on this deployment" in c.value for c in app.caption)


def test_case_note_is_written_and_shown(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(agent, "explain", lambda case: f"Memo for {case['invoice_id']} costing $5.")
    st.cache_data.clear()

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    button(at, "Write case note with Claude").click().run()

    assert not at.exception
    assert at.info[0].value == "Memo for INV-43012 costing \\$5."
