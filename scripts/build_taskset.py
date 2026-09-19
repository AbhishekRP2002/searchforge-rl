"""Build a task mixture from published search benchmarks.

    uv run --with datasets python -m scripts.build_taskset --out searchforge/data/tasks_v1.jsonl

The questions are borrowed; the *distribution* is the contribution. Every row
carries `source_dataset`, so any result can be sliced by where the question came
from — which is the difference between "this provider is better" and "this
provider is better on multi-hop Wikipedia questions".

Honest labelling rule: derive an axis only where the source actually determines
it. `hops` and `answer_type` are derivable. `popularity` and `evidence_depth`
are not -- they need an external signal or a pilot run -- so they are written as
"unknown" rather than guessed. A fabricated axis is worse than a missing one,
because the analysis would slice on it and believe the result.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path

BROWSECOMP_CSV = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)

UNKNOWN = "unknown"
"""Written wherever the source does not determine an axis. Never a guess."""

_MONTH = (
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
)
_DATE = re.compile(
    r"^(?:"
    r"\d{4}-\d{2}-\d{2}"          # 2025-06-05
    rf"|\d{{1,2}}\s+{_MONTH}\s+\d{{4}}"   # 27 Nov 2023
    rf"|{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"  # November 27, 2023
    rf"|{_MONTH}\s+\d{{4}}"                # October 1922
    r"|\d{4}"                       # 1889
    r")$",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"^[-+]?[\d,]+(\.\d+)?\s*(%|[a-zA-Z]{1,12})?$")


def classify_answer(answer: str) -> str:
    """entity | number | date | list, from the answer's own shape.

    Order matters: dates are checked before numbers, because a bare four-digit
    value is a year rather than a count.
    """
    if isinstance(answer, list):
        return "list"
    # Benchmarks punctuate inconsistently -- "7 years." and "7 years" are the
    # same answer, and a trailing period otherwise falls through to entity.
    text = re.sub(r"\s+", " ", str(answer)).strip().rstrip(".")
    if ";" in text or re.search(r",\s*(and\s+)?\w+\s*,", text):
        return "list"
    if _DATE.match(text):
        return "date"
    if _NUMBER.match(text):
        return "number"
    return "entity"


def row(
    *,
    source_dataset: str,
    source_id: str,
    question: str,
    answer: str,
    hops: int,
    recency: str = UNKNOWN,
    answer_type: str | None = None,
    popularity: str = UNKNOWN,
    evidence_depth: str = UNKNOWN,
) -> dict:
    return {
        "source_id": f"{source_dataset}:{source_id}",
        "question": question.strip(),
        "answer": answer,
        "hops": hops,
        "recency": recency,
        "answer_type": answer_type or classify_answer(answer),
        "popularity": popularity,
        "evidence_depth": evidence_depth,
        "source_dataset": source_dataset,
        # Spec section 9 requires a live re-validation stamp. These rows are
        # borrowed as published and have not been re-validated, so the field is
        # empty rather than carrying today's date and implying that they were.
        "validated_on": "",
    }


# --- per-source adapters -----------------------------------------------------
# Each returns an iterable of rows. Wikipedia-derived multi-hop sets are marked
# recency=static because they are built from a frozen encyclopedia dump; the
# open-web sets get UNKNOWN because their questions vary.

SIMPLEQA_ANSWER_TYPE = {"date": "date", "number": "number"}


def load_simpleqa(n, rng):
    from datasets import load_dataset

    data = load_dataset("basicv8vc/SimpleQA", split="test")
    for item in sample(data, n, rng):
        metadata = item["metadata"]
        if isinstance(metadata, str):
            metadata = eval(metadata)  # the column ships as a repr'd dict
        native = str(metadata.get("answer_type", "")).lower()
        yield row(
            source_dataset="simpleqa",
            source_id=hashlib.sha1(item["problem"].encode()).hexdigest()[:12],
            question=item["problem"],
            answer=item["answer"],
            hops=1,
            answer_type=SIMPLEQA_ANSWER_TYPE.get(native, "entity"),
        )


def load_frames(n, rng):
    from datasets import load_dataset

    data = load_dataset("google/frames-benchmark", split="test")
    for item in sample(data, n, rng):
        links = [x for x in (item.get("wiki_links") or []) if x]
        yield row(
            source_dataset="frames",
            source_id=str(item["Unnamed: 0"]),
            question=item["Prompt"],
            answer=item["Answer"],
            # One Wikipedia article per constraint; 3+ is the spec's top bucket.
            hops=min(max(len(links), 2), 3),
        )


def load_hotpotqa(n, rng):
    from datasets import load_dataset

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    for item in sample(data, n, rng):
        if str(item["answer"]).lower() in {"yes", "no"}:
            continue  # a yes/no answer grades as a coin flip, not retrieval
        yield row(
            source_dataset="hotpotqa",
            source_id=item["id"],
            question=item["question"],
            answer=item["answer"],
            hops=2,
            recency="static",
        )


def load_musique(n, rng):
    from datasets import load_dataset

    data = load_dataset("dgslibisey/MuSiQue", split="validation")
    for item in sample(data, n, rng):
        if not item.get("answerable", True):
            continue
        # ids look like "2hop__460946_294723" / "3hop1__..." / "4hop3__..."
        match = re.match(r"(\d+)hop", item["id"])
        yield row(
            source_dataset="musique",
            source_id=item["id"],
            question=item["question"],
            answer=item["answer"],
            hops=min(int(match.group(1)) if match else 2, 3),
            recency="static",
        )


def load_2wiki(n, rng):
    from datasets import load_dataset

    data = load_dataset("voidful/2WikiMultihopQA", split="validation")
    for item in sample(data, n, rng):
        if str(item["answer"]).lower() in {"yes", "no"}:
            continue
        evidences = item.get("evidences") or []
        yield row(
            source_dataset="2wikimultihopqa",
            source_id=item["_id"],
            question=item["question"],
            answer=item["answer"],
            hops=min(max(len(evidences), 2), 3),
            recency="static",
        )


def _browsecomp_decrypt(payload: str, password: str) -> str:
    """OpenAI publishes BrowseComp XOR-encrypted against a per-row canary, so
    the questions are not scrapeable as plain text. Their `derive_key`: SHA-256
    of the password, repeated to the ciphertext length."""
    encrypted = base64.b64decode(payload)
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(encrypted) // len(digest)) + digest[
        : len(encrypted) % len(digest)
    ]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def load_browsecomp(n, rng):
    with urllib.request.urlopen(BROWSECOMP_CSV, timeout=60) as response:
        body = response.read().decode()
    records = list(csv.DictReader(io.StringIO(body)))
    for index, item in sample(records, n, rng, indexed=True):
        yield row(
            source_dataset="browsecomp",
            source_id=str(index),
            question=_browsecomp_decrypt(item["problem"], item["canary"]),
            answer=_browsecomp_decrypt(item["answer"], item["canary"]),
            hops=3,  # multi-hop by construction; 3 is the spec's "3+" bucket
        )


SOURCES = {
    "simpleqa": load_simpleqa,
    "frames": load_frames,
    "hotpotqa": load_hotpotqa,
    "musique": load_musique,
    "2wikimultihopqa": load_2wiki,
    "browsecomp": load_browsecomp,
}

DEFAULT_COUNTS = {
    "simpleqa": 60,
    "frames": 60,
    "hotpotqa": 60,
    "musique": 60,
    "2wikimultihopqa": 60,
    # Off by default, for two independent reasons.
    #
    # Distribution: OpenAI publishes BrowseComp encrypted precisely so the
    # questions stay out of training corpora and web crawls. Writing decrypted
    # rows into a file committed to a public repository would undo that and
    # help contaminate the benchmark for everyone. Generate them locally with
    # `--browsecomp N` and keep the output untracked.
    #
    # Signal: it is built so answers are very hard to find. At a 10-call budget
    # most rollouts fail, and a group whose rewards are all zero yields no GRPO
    # advantage -- full cost, no gradient. It is a headroom probe, not training
    # signal, so keep any local sample small.
    "browsecomp": 0,
}


def sample(data, n, rng, *, indexed=False):
    """Deterministic oversample. Adapters drop rows, so take more than asked."""
    order = list(range(len(data)))
    rng.shuffle(order)
    picked = order[: min(len(order), int(n * 1.5) + 10)]
    return [(i, data[i]) for i in picked] if indexed else [data[i] for i in picked]


def build(counts: dict[str, int], seed: int) -> list[dict]:
    rows: list[dict] = []
    for name, wanted in counts.items():
        if not wanted:
            continue
        produced = []
        for item in SOURCES[name](wanted, random.Random(seed)):
            produced.append(item)
            if len(produced) >= wanted:
                break
        print(f"  {name:18} {len(produced):4} rows")
        rows.extend(produced)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("searchforge/data/tasks_v1.jsonl"))
    parser.add_argument("--seed", type=int, default=0, help="deterministic sampling")
    for name, default in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()

    counts = {name: getattr(args, name.replace("-", "_")) for name in SOURCES}
    print(f"building into {args.out} (seed {args.seed})")
    rows = build(counts, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
