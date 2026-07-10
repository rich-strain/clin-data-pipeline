"""Stage D — train/val split of Stage C's curated output.

Takes `data/curated/synthesized.jsonl` (Stage C's final output, currently
132 records at the 100-patient scale) and partitions it into
`data/curated/split_train.jsonl` / `data/curated/split_val.jsonl` — still
in the curated record shape (patient_id/diagnoses/medications/vitals/...),
not yet the instruction/response training format. `format_jsonl.py` reads
these two files next and does the note-lookup + redaction + instruction/
response formatting into `data/splits/train.jsonl` / `data/splits/
val.jsonl`. Splitting and formatting are kept as separate steps so the
"which record goes in which split" decision and the "how a record becomes
a training example" decision can each be inspected/verified independently.

**Excluding any `synthesize.py` records.** Per CLAUDE.md's Resolved
decisions #8, these records (tagged `"synthesized": true`) have no matching
entry in `data/generated/clinical_notes.jsonl` — they were fabricated
directly as structured fields, bypassing note generation entirely. Since
every other record's training instruction is built from real note text
(see `format_jsonl.py`), giving these a different instruction shape (e.g.
a synthesized prompt instead of note text) would make the instruction
format inconsistent across the training set. Excluded here rather than
force-fit. At the current 100-patient scale, `synthesize.py` found every
diagnosis category already represented after rebalancing and synthesized
zero new records, so this exclusion is currently a no-op (132 curated
records -> 132 eligible for Stage D) — but the exclusion logic stays in
place since it isn't guaranteed to stay at zero on a future regeneration.

**~80/20 split, but grouped by original patient identity, not by raw
record.** `rebalance.py` produces duplicate records (via
`rebalance_duplicate_of`) that are near-identical copies of an existing
record, just to correct diagnosis-category representation — they are not
independent patients. If a duplicate landed in val while its original sat
in train, val would effectively contain content the model already saw
during training almost verbatim, silently inflating the validation metric
into meaninglessness. So the unit of splitting here is the **original
patient group** (a patient's original record plus every `-dupN` copy of
it), not the individual record: every record in a group goes to the same
split, always. At the current 100-patient scale this collapses 132
eligible records into 100 groups (group sizes vary more widely than the
original 10-patient dev sample — most patients are a group of 1, but
categories that needed heavier rebalancing produce groups as large as 10);
an 80/20 split *on groups* gives 80 train groups / 20 val groups.

**Consequence, stated honestly:** because group sizes vary, splitting by
group only *approximately* hits an 80/20 *record* ratio, not exactly — at
the current scale it lands at 112/132 train (~85%) vs. 20/132 val (~15%),
because the larger duplicate groups happen to have landed in train this
run (see verification output for the live numbers on any given run).
That's an intentional trade-off: correctness of the anti-leakage grouping
constraint takes priority over hitting an exact record-count ratio.

**Ordering:** groups are assigned to train/val in first-seen order from
`synthesized.jsonl` (train = first 80% of groups encountered, val = the
rest) rather than an additional random shuffle. Patient UUIDs were already
randomly generated in Stage A, so the file's existing order carries no
structure to correct for — adding a second RNG pass here would just be
another seed to document for no real benefit, so this stays fully
deterministic with zero randomness.
"""

import argparse
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHESIZED_PATH = DATA_DIR / "curated" / "synthesized.jsonl"
SPLIT_TRAIN_PATH = DATA_DIR / "curated" / "split_train.jsonl"
SPLIT_VAL_PATH = DATA_DIR / "curated" / "split_val.jsonl"

TRAIN_FRACTION = 0.8


def original_patient_id(record):
    """The original patient a record belongs to — itself, unless it's a
    rebalance.py duplicate, in which case it's whoever that duplicate was
    copied from."""
    return record.get("rebalance_duplicate_of", record["patient_id"])


def read_records(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def group_by_original_patient(records):
    """Return an ordered dict: original_patient_id -> [records], in
    first-seen order."""
    groups = {}
    for r in records:
        groups.setdefault(original_patient_id(r), []).append(r)
    return groups


def split_groups(groups):
    """Return (train_records, val_records), split by whole group."""
    group_keys = list(groups)
    n_train_groups = math.ceil(len(group_keys) * TRAIN_FRACTION)
    train_keys = group_keys[:n_train_groups]
    val_keys = group_keys[n_train_groups:]

    train_records = [r for k in train_keys for r in groups[k]]
    val_records = [r for k in val_keys for r in groups[k]]
    return train_records, val_records, train_keys, val_keys


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split curated records into train/val, grouped by original patient.")
    parser.add_argument("--in", dest="in_path", type=Path, default=SYNTHESIZED_PATH)
    parser.add_argument("--train-out", type=Path, default=SPLIT_TRAIN_PATH)
    parser.add_argument("--val-out", type=Path, default=SPLIT_VAL_PATH)
    args = parser.parse_args()

    all_records = list(read_records(args.in_path))
    synthesized = [r for r in all_records if r.get("synthesized")]
    eligible = [r for r in all_records if not r.get("synthesized")]

    groups = group_by_original_patient(eligible)
    train_records, val_records, train_keys, val_keys = split_groups(groups)

    write_jsonl(train_records, args.train_out)
    write_jsonl(val_records, args.val_out)

    print(f"{len(all_records)} curated records ({len(synthesized)} synthesized, excluded); {len(eligible)} eligible")
    print(f"{len(groups)} original patient groups -> {len(train_keys)} train / {len(val_keys)} val")
    print(f"Wrote {len(train_records)} records to {args.train_out}")
    print(f"Wrote {len(val_records)} records to {args.val_out}")

    # Verify: no original patient group split across train/val.
    train_patient_set = {original_patient_id(r) for r in train_records}
    val_patient_set = {original_patient_id(r) for r in val_records}
    overlap = train_patient_set & val_patient_set
    if overlap:
        print(f"LEAKAGE: {len(overlap)} patient group(s) appear in both splits: {sorted(overlap)}")
    else:
        print("Verified: no original patient group appears in both train and val.")


if __name__ == "__main__":
    main()
