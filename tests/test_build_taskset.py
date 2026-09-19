"""Checks for the taskset builder's two pieces of real logic.

`scripts.build_taskset` imports `datasets` lazily inside each loader, so this
module imports without that dependency installed.
"""

import pytest

from scripts.build_taskset import UNKNOWN, classify_answer, row


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("2025-06-05", "date"),
        ("27 Nov 2023", "date"),
        ("November 27, 2023", "date"),
        ("October 1922", "date"),       # month + year, no day
        ("1889", "date"),               # a bare year is a date, not a count
        ("7 years.", "number"),         # benchmarks punctuate inconsistently
        ("56,000", "number"),
        ("42", "number"),
        ("3.11", "number"),
        ("Fusajiro Yamauchi", "entity"),
        ("Hey Pa! There's a Goat on the Roof", "entity"),
        ("red; green; blue", "list"),
        ("red, green, blue", "list"),
        (["a", "b"], "list"),
    ],
)
def test_classify_answer(answer, expected):
    assert classify_answer(answer) == expected


def test_row_namespaces_source_id_and_never_fabricates_an_axis():
    built = row(
        source_dataset="hotpotqa",
        source_id="abc123",
        question="  Who founded Nintendo?  ",
        answer="Fusajiro Yamauchi",
        hops=2,
        recency="static",
    )

    # source_id is namespaced so ids from different benchmarks cannot collide
    # when the mixture is joined on it for paired analysis.
    assert built["source_id"] == "hotpotqa:abc123"
    assert built["source_dataset"] == "hotpotqa"
    assert built["question"] == "Who founded Nintendo?"
    assert built["recency"] == "static"
    # Axes the source does not determine stay unknown rather than guessed.
    assert built["popularity"] == UNKNOWN
    assert built["evidence_depth"] == UNKNOWN
    # Spec section 9 wants a live re-validation stamp; these rows have not had one.
    assert built["validated_on"] == ""


def test_browsecomp_decrypt_round_trips():
    import base64
    import hashlib

    from scripts.build_taskset import _browsecomp_decrypt

    plaintext, password = "Who founded Nintendo?", "canary-xyz"
    digest = hashlib.sha256(password.encode()).digest()
    raw = plaintext.encode()
    key = digest * (len(raw) // len(digest)) + digest[: len(raw) % len(digest)]
    payload = base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode()

    assert _browsecomp_decrypt(payload, password) == plaintext
