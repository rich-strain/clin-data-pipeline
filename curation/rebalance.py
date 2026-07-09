"""Stage C — rebalancing of Stage C's redacted records.

Takes `data/curated/redacted.jsonl` (Stage C's redact output) and evens out
an under-represented axis by duplicating existing records, rather than
generating new content — that's deliberately left to `synthesize.py`, the
next and final curation sub-step.

**Which axis, and why:** looking at the actual 10-record dev sample rather
than assuming, three candidate axes exist — diagnosis category counts,
medication category counts, and per-record vitals count. Medication counts
are comparatively tight (1-4 mentions per medication across 10 records) and
vitals-per-record is a field-density issue, not a class-imbalance one.
Diagnosis category is the clearest skew (`Hypothyroidism, unspecified`
appears in 3/10 records while `Essential (primary) hypertension`,
`Migraine, unspecified, not intractable`, and `Type 2 diabetes mellitus
without complications` each appear in only 1/10), and it's the axis most
directly analogous to a class-imbalance problem for a diagnosis-extraction
fine-tune: a model trained on this data would see 3x more hypothyroidism
examples than hypertension ones, for no reason related to real-world
prevalence — it's an artifact of `random.choice` over a 10-condition list
at `n_patients=10`. So diagnosis category (counted once per record, not
per raw mention — a record listing the same diagnosis twice, which
`normalize.py` already dedupes, shouldn't count double) is the axis this
module rebalances.

**Decision: oversample (duplicate existing records), don't filter
(downsample over-represented ones).** With only 10 records, downsampling to
match the rarest category would shrink an already-tiny dataset even
further — actively worse for a fine-tune that's already data-starved.
Duplicating records that carry an under-represented diagnosis preserves
every original example while boosting rare categories. The cost:
duplicated records are exact copies (same wording, same vitals), so this
risks the model overfitting to those specific duplicated examples rather
than genuinely learning the rare category better. That's precisely why
this technique is a stopgap, not the real fix — `synthesize.py` (next)
generates *new*, differently-worded records for the gaps that duplication
alone can't legitimately fill (see below).

**Duplicated records are marked, not silently blended in.** Each duplicate
gets `"rebalance_duplicate_of": "<original patient_id>"` and a suffixed
`patient_id` (`<original>-dup1`, `-dup2`, ...) — both for auditability here,
and because it matters downstream: **Stage D's train/val split must keep a
duplicated record in the same split as its original**, or a near-identical
example leaking across the split boundary would artificially inflate
validation performance. Flagging this now so it isn't missed when `split.py`
is built.

**Known limitation — multi-diagnosis records cause overshoot.** A record
with two diagnoses boosts both when duplicated, even if only one was
actually deficient. On this sample, duplicating a record to fix
`Type 2 diabetes mellitus without complications` (1 -> 3) also carries
`Unspecified asthma, uncomplicated` along for the ride, pushing asthma from
2 to 6 — well past the target. A smarter set-cover-style selection could
minimize this, but that's not worth building for a 10-record dev sample;
documented here rather than silently accepted or over-engineered away.

**Known limitation — zero-represented categories can't be rebalanced at
all.** `Hyperlipidemia, unspecified` appears in *zero* of the 10 records.
There's no existing record containing it to duplicate — oversampling can
only amplify what's already present, never manufacture a category from
nothing. This is exactly the gap `synthesize.py` exists to fill.

**Honesty about scale:** with 10 records, this isn't statistically
meaningful rebalancing — it's a deterministic demonstration of the
technique (and its trade-offs) at toy scale, not a claim that the resulting
distribution is well-calibrated. On a realistically-sized dataset, exact
duplication would still carry the same overfitting risk per duplicate;
a real pipeline would lean on `synthesize.py`-style generation for
oversampling once record counts make paraphrased regeneration practical,
rather than scaling this module's verbatim-duplication approach up as-is.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.generate_fhir import CONDITIONS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REDACTED_PATH = DATA_DIR / "curated" / "redacted.jsonl"
REBALANCED_PATH = DATA_DIR / "curated" / "rebalanced.jsonl"

CANONICAL_CONDITION_ORDER = [display for _, display in CONDITIONS]


def _diagnosis_names(record):
    return {d["name"] for d in record["diagnoses"]}


def category_counts(records):
    """Number of records containing each diagnosis category (dedup'd per record)."""
    counts = collections.Counter()
    for r in records:
        counts.update(_diagnosis_names(r))
    return counts


def _pick_templates(records):
    """First record (in original order) containing each category, if any."""
    templates = {}
    for category in CANONICAL_CONDITION_ORDER:
        for r in records:
            if category in _diagnosis_names(r):
                templates[category] = r
                break
    return templates


def rebalance_records(records):
    """Return (augmented_records, duplicates_added: list of dup records)."""
    before_counts = category_counts(records)
    target = max(before_counts.values()) if before_counts else 0
    templates = _pick_templates(records)

    augmented = list(records)
    duplicates = []
    dup_index = collections.Counter()

    for category in CANONICAL_CONDITION_ORDER:
        template = templates.get(category)
        if template is None:
            continue  # zero-represented category: nothing to duplicate, see docstring
        current = sum(1 for r in augmented if category in _diagnosis_names(r))
        while current < target:
            dup_index[template["patient_id"]] += 1
            dup = dict(template)
            dup["patient_id"] = f"{template['patient_id']}-dup{dup_index[template['patient_id']]}"
            dup["rebalance_duplicate_of"] = template["patient_id"]
            augmented.append(dup)
            duplicates.append(dup)
            current += 1

    return augmented, duplicates


def read_records(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Rebalance diagnosis-category representation in redacted records.")
    parser.add_argument("--in", dest="in_path", type=Path, default=REDACTED_PATH)
    parser.add_argument("--out", type=Path, default=REBALANCED_PATH)
    args = parser.parse_args()

    records = list(read_records(args.in_path))
    before_counts = category_counts(records)
    augmented, duplicates = rebalance_records(records)
    after_counts = category_counts(augmented)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in augmented:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(augmented)} records ({len(records)} original + {len(duplicates)} duplicates) to {args.out}")
    print(f"\n{'diagnosis category':55} {'before':>7} {'after':>7}")
    for category in CANONICAL_CONDITION_ORDER:
        b = before_counts.get(category, 0)
        a = after_counts.get(category, 0)
        flag = "  <- still 0, needs synthesize.py" if a == 0 else ""
        print(f"{category:55} {b:>7} {a:>7}{flag}")

    if duplicates:
        print("\nduplicates added:")
        for d in duplicates:
            print(f"  {d['patient_id']} (dup of {d['rebalance_duplicate_of']}): {sorted(_diagnosis_names(d))}")


if __name__ == "__main__":
    main()
