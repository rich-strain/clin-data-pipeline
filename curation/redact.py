"""Stage C — redaction of synthetic PHI/PII fields from normalized records.

Takes `data/curated/normalized.jsonl` (Stage C's normalize output) and
handles the PHI-standin fields `extractor.py` was deliberately built to
capture as concrete structured fields for this step to act on (see that
module's docstring): `patient_name`, `mrn`, `address`, `date_of_birth`,
`note_date` (visit date). `mrn` and `address` were added in a later pass
(CLAUDE.md Resolved decisions #9) — `generate_fhir.py` had generated a
synthetic MRN from the start, but it never flowed downstream into notes,
extraction, or curated records until then.

**Two different actions for two different kinds of field.**

- `patient_name`, `mrn`, and `address` carry no analytic value for a
  diagnosis-extraction model — all three are dropped entirely, same
  reasoning: nothing downstream benefits from a masked-but-present value,
  and a placeholder token is just something the model has to learn to
  ignore. `mrn`/`address` are `None` on some records already (messy
  generation can drop either from the Patient resource) — stripping the
  key works the same whether the value is a string or already `None`, so
  no special-casing is needed for that.

- `date_of_birth` and `note_date` are **date-shifted, not stripped or
  dropped.** Dates carry clinical/temporal meaning a real fine-tune could
  reasonably want (e.g. the relationship between DOB and visit date, or
  intervals between visits, are exactly the kind of signal a model
  extracting "patient age at time of visit" would need) — deleting them
  destroys that. But leaving true dates intact makes re-identification
  meaningfully easier (dates are one of HIPAA Safe Harbor's 18 identifying
  fields), so each date is shifted by a random offset instead of being
  either kept verbatim or removed.

  The shift is seeded **per patient, per category** — not per field. There
  are two categories: `dob` (just `date_of_birth`) and `visit` (every
  visit/procedure-type date field for that patient — currently just
  `note_date`, but the seeding key is the category name, not the field
  name, specifically so a second visit-type date field added later (e.g. a
  medication-authored date, if extraction ever captures one) shares
  `note_date`'s `visit_shift` rather than silently getting its own
  independent offset. Seeding per-field instead of per-category would look
  identical to this today, with only one field per category, and then
  quietly break the moment a second visit-type field showed up — that's
  exactly the bug the first version of this module had.

  Offsets are derived from `sha256(patient_id + category name)` seeding a
  `random.Random` (so a run is reproducible without persisting a separate
  offset table). `dob_shift` and `visit_shift` are independent per patient —
  a single shared offset per patient would let anyone who recovers one true
  date (e.g. a known birthdate) trivially unshift every other date for that
  patient by simple subtraction; independent category shifts mean
  recovering one category's date reveals nothing about the other's.

  **Shift range: +/- 365 days.** Large enough that the shifted date isn't
  trivially close to the real one (ruling out "off by a day or two"
  guessing), small enough to keep the date roughly in the right season/year
  for anyone eyeballing the data for plausibility. This is a portfolio-scale
  choice, not a HIPAA Safe Harbor compliance claim — Safe Harbor requires
  dates be generalized to year only for anyone 90+; this dataset's ages
  don't reach that threshold, so day-level shifting is sufficient here.

**Scope note:** this module only redacts the notes -> extraction ->
curation branch. `data/generated/patient_features.csv` (Stage A's
flattened FHIR feature table) is a separate, never-downstream-consumed
artifact and is intentionally not touched here — see
`docs/design_decisions.md` for why, and for the app-labeling action item
that scope decision requires.
"""

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NORMALIZED_PATH = DATA_DIR / "curated" / "normalized.jsonl"
REDACTED_PATH = DATA_DIR / "curated" / "redacted.jsonl"

STRIPPED_FIELDS = ("patient_name", "mrn", "address")

# field -> shift category. All fields in the same category share one offset
# per patient. "visit" covers every visit/procedure-type date field; today
# that's only note_date, but a second visit-type field (e.g. a future
# medication-authored date) belongs in this same category, not a new one.
SHIFTED_FIELDS = {
    "date_of_birth": "dob",
    "note_date": "visit",
}
SHIFT_RANGE_DAYS = 365


def _shift_offset_days(patient_id, category):
    """Deterministic per-(patient, category) offset in [-SHIFT_RANGE_DAYS, SHIFT_RANGE_DAYS]."""
    seed = int(hashlib.sha256(f"{patient_id}:{category}".encode("utf-8")).hexdigest(), 16)
    return random.Random(seed).randint(-SHIFT_RANGE_DAYS, SHIFT_RANGE_DAYS)


def shift_date(patient_id, category, date_str):
    if date_str is None:
        return None
    d = date.fromisoformat(date_str)
    offset = _shift_offset_days(patient_id, category)
    return (d + timedelta(days=offset)).isoformat()


def redact_record(record):
    """Return (redacted_record, audit: dict of {field: {"action", "original"}})."""
    redacted = dict(record)
    patient_id = redacted.get("patient_id")
    audit = {}

    for field in STRIPPED_FIELDS:
        if field in redacted:
            audit[field] = {"action": "stripped", "original": redacted.pop(field)}

    for field, category in SHIFTED_FIELDS.items():
        if field in redacted:
            original = redacted[field]
            redacted[field] = shift_date(patient_id, category, original)
            audit[field] = {"action": "shifted", "original": original, "category": category}

    return redacted, audit


def read_records(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Redact PHI/PII fields from normalized extraction records.")
    parser.add_argument("--in", dest="in_path", type=Path, default=NORMALIZED_PATH)
    parser.add_argument("--out", type=Path, default=REDACTED_PATH)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    field_counts = {field: 0 for field in STRIPPED_FIELDS + tuple(SHIFTED_FIELDS)}
    with args.out.open("w") as f:
        for record in read_records(args.in_path):
            redacted, audit = redact_record(record)
            f.write(json.dumps(redacted) + "\n")
            count += 1
            for field in audit:
                field_counts[field] += 1

    print(f"Wrote {count} redacted records to {args.out}")
    for field in STRIPPED_FIELDS:
        print(f"  stripped {field!r} from {field_counts[field]}/{count} records")
    for field in SHIFTED_FIELDS:
        print(f"  date-shifted {field!r} (+/- {SHIFT_RANGE_DAYS} days) on {field_counts[field]}/{count} records")


if __name__ == "__main__":
    main()
