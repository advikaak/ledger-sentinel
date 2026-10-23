"""
Check that a case note only states things its evidence contains.

The explanation stage asks Claude to narrate a decision the model already made. The
system prompt says "cite the specific records" and "never re-judge", but a prompt is a
request, not a guarantee. This module is the guarantee we can actually test: it pulls
every invoice number, vendor number, account suffix, dollar amount and other figure out
of a note and confirms each one appears in the case the note was written from.

What this checks: that the note's FACTS are in the evidence.
What it does not check: that the note is well written, that its reasoning is sound, or
that it picked the most useful first check for the analyst. A note can pass this check
and still be unhelpful. Those need a human reader (or a labelled set of notes), and
this project has not measured them.
"""

import json
import re

INVOICE_ID = re.compile(r"\bINV-\d+(?:-[A-Z0-9]+)?\b")
VENDOR_ID = re.compile(r"\bV\d{4}\b")
ACCOUNT_SUFFIX = re.compile(r"\*{4}(\d{4})")
# A number as people write it: $1,234.56  181k  0.98  98%  5
NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s?(?:[kK]\b|%)?")


def _flatten(obj):
    """Every string and number inside a nested case, as plain text and as floats."""
    texts, numbers = [], []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                texts.append(str(k))
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
        elif isinstance(x, bool) or x is None:
            return
        elif isinstance(x, (int, float)):
            numbers.append(float(x))
        else:
            texts.append(str(x))

    walk(obj)
    return texts, numbers


def _allowed_numbers(case):
    """
    Numbers a faithful note may quote: anything in the case, its absolute value, a
    fraction written as a percentage, and the gaps between two quoted amounts (a note
    may say "$80 more than" without that figure being stored).
    """
    texts, numbers = _flatten(case)
    # Digits that appear inside strings, e.g. the 1234 in "****1234" or the 40231 in INV-40231.
    for t in texts:
        numbers.extend(float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", t))
    allowed = set()
    for n in numbers:
        allowed.add(abs(n))
        if 0 < abs(n) <= 1:
            allowed.add(abs(n) * 100)
    amounts = [abs(n) for n in numbers if abs(n) >= 100]
    for i, a in enumerate(amounts):
        for b in amounts[i + 1 :]:
            allowed.add(abs(a - b))
    return allowed


def _to_float(token):
    t = token.replace("$", "").replace(",", "").strip()
    scale = 1.0
    if t.lower().endswith("k"):
        scale, t = 1000.0, t[:-1]
    t = t.rstrip("% ").strip()
    return float(t) * scale if t else None


def _close(value, allowed):
    """Rounding is fine ("$25,280" for $25,280.48); invention is not."""
    tolerance = 0.51 if value < 1000 else value * 0.005
    return any(abs(value - a) <= tolerance for a in allowed)


def ungrounded_claims(note, case):
    """
    Everything in `note` that the case does not support, as a list of strings.
    Empty list means every identifier and figure in the note is in the evidence.
    """
    texts, _ = _flatten(case)
    haystack = " ".join(texts)
    problems = []

    for pattern, kind in ((INVOICE_ID, "invoice"), (VENDOR_ID, "vendor")):
        for token in sorted(set(pattern.findall(note))):
            if token not in haystack:
                problems.append(f"{kind} {token} is not in the evidence")

    for suffix in sorted(set(ACCOUNT_SUFFIX.findall(note))):
        if suffix not in haystack:
            problems.append(f"account ending {suffix} is not in the evidence")

    # Identifiers were checked above; remove them so their digits aren't read as figures.
    stripped = INVOICE_ID.sub(" ", note)
    stripped = VENDOR_ID.sub(" ", stripped)
    stripped = ACCOUNT_SUFFIX.sub(" ", stripped)
    allowed = _allowed_numbers(case)
    for token in NUMBER.findall(stripped):
        value = _to_float(token)
        if value is None or (value < 10 and "$" not in token and "%" not in token):
            # Counts like "one" or "two" days are prose; single digits are too ambiguous
            # to police without flagging ordinary sentences.
            continue
        if not _close(value, allowed):
            problems.append(f"figure {token.strip()} is not in the evidence")
    return problems


if __name__ == "__main__":
    import sys

    case_path, note_path = sys.argv[1:3]
    case = json.load(open(case_path))
    problems = ungrounded_claims(open(note_path).read(), case)
    print("\n".join(problems) if problems else "every identifier and figure is in the evidence")
