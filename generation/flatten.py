"""Projects Stage A's FHIR bundles into a flat, one-row-per-patient feature table.

This is a *view* derived from the FHIR bundles (the system of record), not a
separate source of truth — see CLAUDE.md's rationale for why FHIR is
generated first here rather than a converted-afterward export.

Conditions, medications, and observations are all multi-valued per patient,
but a classical-ML feature table needs one row per patient with a fixed set
of columns. The collapsing strategy here is a real design decision:

- Conditions / medications: collapsed into a count column plus a
  semicolon-joined summary column, rather than one-hot columns per code.
  One-hot would be more ML-ready but blows up the column count for an
  open-ended, growing code list; a summary column keeps the table readable
  for this portfolio's purposes and can be one-hot encoded later if needed.
- Observations: collapsed to "most recent value" per vital/lab, since a
  single snapshot per patient is what most flat feature tables use latest
  labs for. "Most recent" is picked by effectiveDateTime.

  Resolution of the missing-date gap: when the messiness toggle drops
  effectiveDateTime, dropping the reading (rather than the value) is the
  intended messiness — a real EHR feed does have observations with no
  reliable timestamp, and that's a legitimate problem for curation to
  surface, not paper over at generation time. So the fix belongs here in
  flatten, not in the generator: when none of a vital's readings have a
  date, flatten still picks a value (falling back to generation order,
  the best available signal) but marks it with a companion
  `{slug}_date_unknown` boolean column, so "most recent" is never silently
  presented as reliable when it isn't. Downstream consumers (Stage C, or a
  model) can then decide whether to trust or drop that value explicitly.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from fhir_common import group_bundle_entries

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

OBSERVATION_DISPLAYS = [
    "Heart rate",
    "Systolic blood pressure",
    "Body temperature",
    "Body weight",
    "Body height",
    "Glucose",
]


def _parse_date(value):
    return date.fromisoformat(value) if value else None


def _slug(display):
    return display.lower().replace(" ", "_")


def flatten_bundle(bundle):
    patient, conditions, medications, observations = group_bundle_entries(bundle)

    name = patient.get("name", [{}])[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")

    all_dates = [c["onset"] for c in conditions if c["onset"]]
    all_dates += [m["effective"] for m in medications if m["effective"]]
    all_dates += [o["date"] for readings in observations.values() for o in readings if o["date"]]
    as_of = max((_parse_date(d) for d in all_dates), default=None)
    birth_date = _parse_date(patient["birthDate"])

    age_at_last_encounter = None
    if as_of and birth_date:
        had_birthday = (as_of.month, as_of.day) >= (birth_date.month, birth_date.day)
        age_at_last_encounter = as_of.year - birth_date.year - (0 if had_birthday else 1)

    row = {
        "patient_id": patient["id"],
        "mrn": patient.get("identifier", [{}])[0].get("value", ""),
        "given_name": given,
        "family_name": family,
        "gender": patient.get("gender", ""),
        "birth_date": patient["birthDate"],
        "age_at_last_encounter": age_at_last_encounter,
        "condition_count": len(conditions),
        "conditions": "; ".join(
            f"{c['display']} ({c['onset'] or 'unknown date'})" for c in conditions
        ),
        "medication_count": len(medications),
        "medications": "; ".join(
            f"{m['display']} [{m['dosage_text'] or 'no dosage recorded'}]" for m in medications
        ),
    }

    for display in OBSERVATION_DISPLAYS:
        slug = _slug(display)
        readings = observations.get(display, [])
        dated = [r for r in readings if r["date"]]
        if dated:
            latest = max(dated, key=lambda r: r["date"])
            date_unknown = False
        else:
            latest = readings[-1] if readings else None
            date_unknown = latest is not None
        row[f"{slug}_value"] = latest["value"] if latest else None
        row[f"{slug}_unit"] = latest["unit"] if latest else None
        row[f"{slug}_date_unknown"] = date_unknown if latest else None

    return row


def flatten_dataset(bundles):
    return pd.DataFrame([flatten_bundle(b) for b in bundles])


def main():
    parser = argparse.ArgumentParser(description="Flatten FHIR bundles into a patient feature table.")
    parser.add_argument("--in", dest="in_path", type=Path, default=DATA_DIR / "fhir_bundles.json")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "patient_features.csv")
    args = parser.parse_args()

    bundles = json.loads(args.in_path.read_text())
    df = flatten_dataset(bundles)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.out}")


if __name__ == "__main__":
    main()
