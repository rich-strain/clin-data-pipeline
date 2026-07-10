"""Encodes the flattened patient feature table into a scikit-learn-ready matrix.

`flatten.py` produces a human-readable feature table, but it is not directly
consumable by a classical ML model: `gender` is text, `conditions`/
`medications` are semicolon-joined multi-value strings, several vital columns
are missing, and the vital values arrive in mixed units (kg vs lb, C vs F)
courtesy of Stage A's messiness toggle. A scikit-learn estimator needs a fully
numeric matrix with no missing cells and no false ordinal relationships. This
module closes that gap using standard `sklearn.preprocessing` transforms —
turning "flattened for display" into "flattened for machine learning."

- gender          -> OneHotEncoder: one independent 0/1 column per category,
                     rather than an arbitrary integer map (male=0, female=1)
                     that would imply a meaningless ordering the model could
                     wrongly learn along. Missing gender (dropped ~15% of the
                     time by messiness) still zeroes both one-hot columns via
                     handle_unknown="ignore", but that alone is ambiguous —
                     a lone 0 can't be told apart from "confirmed not this
                     category." An explicit `gender_unknown` flag makes
                     "we don't know" its own visible signal instead of
                     something inferred from two other columns being zero,
                     the same pattern flatten.py already uses for
                     `{vital}_date_unknown`.
- conditions      -> MultiLabelBinarizer over the canonical ICD-10 conditions
- medications        / RxNorm medications (imported from generate_fhir.py, the
                     single source of truth). One 0/1 column per canonical
                     entry; a patient can be 1 in several at once (multi-label)
                     — something one-hot encoding cannot represent, since a
                     patient legitimately has several diagnoses simultaneously.
- vital *_value   -> unit-normalized to canonical units first (lb->kg, in->cm,
                     degF->degC, using the unit column flatten emits) so values
                     are comparable across patients, then SimpleImputer (mean)
                     fills the genuinely-missing readings.
- *_date_unknown  -> bool cast to 0/1.
- identifiers     -> patient_id/mrn/name/birth_date dropped (not features;
                     patient_id is re-attached as a leading join key only, not
                     a model input).
- *_unit          -> dropped: after unit normalization these are constant
                     (zero-variance), so they carry no signal.

**Trade-offs (per this project's "explain, don't just pick one" convention):**
- Mean imputation is the simplest defensible default; it distorts the feature
  distribution less than zero-fill but ignores structure in the missingness
  (a vital may be absent *because* it wasn't clinically relevant, not at
  random). A production pipeline might add a missing-indicator column or a
  model-based imputer — out of scope for demonstrating the mechanics.
- Unit normalization is done here rather than trusting the raw values because
  a body-weight column mixing 92 (kg) and 138 (lb) is genuinely garbage as a
  feature; harmonizing units is a real, required part of feature prep, not
  optional polish.

This encodes Stage A's *raw* feature branch, which is never fed to Stage D/E
fine-tuning (that's the separate notes -> extraction branch). It exists to
demonstrate FHIR -> flat table -> ML-ready matrix end to end — the classical-ML
preparation path, the same category of work as a scikit-learn portfolio piece.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.generate_fhir import CONDITIONS, MEDICATIONS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

CONDITION_DISPLAYS = [display for _, display in CONDITIONS]
# MEDICATIONS tuples are (code, display, shorthand, fulltext) — we want display.
MEDICATION_DISPLAYS = [display for _, display, _shorthand, _fulltext in MEDICATIONS]

IDENTIFIER_COLS = ["mrn", "given_name", "family_name", "birth_date"]

# Vital slug -> (canonical unit, alt unit string flatten emits, inverse-to-canonical fn).
# Only vitals with an alternate unit need normalizing; HR/BP/glucose are single-unit.
VITAL_UNIT_NORMALIZERS = {
    "body_temperature": ("Cel", "[degF]", lambda f: round((f - 32) * 5 / 9, 1)),
    "body_weight": ("kg", "[lb_av]", lambda lb: round(lb / 2.20462, 1)),
    "body_height": ("cm", "[in_i]", lambda inches: round(inches * 2.54, 1)),
}
VITAL_SLUGS = [
    "heart_rate",
    "systolic_blood_pressure",
    "body_temperature",
    "body_weight",
    "body_height",
    "glucose",
]


def _parse_multivalue(cell, suffix_open):
    """Split a '; '-joined multi-value cell into its canonical display names.

    Each entry looks like '{display} (onset)' for conditions or
    '{display} [dosage]' for medications; the canonical display is everything
    before the final ' (' / ' [' (no canonical display itself contains those).
    Empty/NaN cells yield an empty list.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    entries = [e.strip() for e in cell.split(";") if e.strip()]
    return [e.rsplit(suffix_open, 1)[0] for e in entries]


def _normalize_vital_value(value, unit, normalizer):
    """Convert a value to canonical units when it arrived in the alt unit."""
    canonical_unit, alt_unit, to_canonical = normalizer
    if pd.isna(value):
        return value
    if isinstance(unit, str) and unit == alt_unit:
        return to_canonical(value)
    return value


def encode(df):
    """Return (encoded_df, summary_dict) — encoded_df is fully numeric except
    for the leading patient_id join key."""
    pieces = [df[["patient_id"]].reset_index(drop=True)]

    # --- gender: one-hot + explicit missing indicator -----------------------
    gender_unknown = df["gender"].isna().reset_index(drop=True)
    gender = df[["gender"]].fillna("__missing__")
    ohe = OneHotEncoder(
        categories=[["female", "male"]],
        handle_unknown="ignore",  # '__missing__' -> all-zero row
        sparse_output=False,
    )
    gender_encoded = ohe.fit_transform(gender)
    gender_cols = [f"gender_{c}" for c in ohe.categories_[0]]
    gender_df = pd.DataFrame(gender_encoded.astype(int), columns=gender_cols)
    gender_df["gender_unknown"] = gender_unknown.astype(int)
    pieces.append(gender_df)

    # --- conditions / medications: multi-label binarize --------------------
    for raw_col, prefix, classes, suffix in (
        ("conditions", "dx", CONDITION_DISPLAYS, " ("),
        ("medications", "rx", MEDICATION_DISPLAYS, " ["),
    ):
        labels = df[raw_col].apply(lambda cell: _parse_multivalue(cell, suffix))
        mlb = MultiLabelBinarizer(classes=classes)
        matrix = mlb.fit_transform(labels)
        cols = [f"{prefix}: {c}" for c in mlb.classes_]
        pieces.append(pd.DataFrame(matrix.astype(int), columns=cols))

    # --- simple numeric passthroughs ---------------------------------------
    pieces.append(df[["condition_count", "medication_count"]].reset_index(drop=True).astype(int))

    # --- vitals: unit-normalize, then mean-impute --------------------------
    value_cols = [f"{slug}_value" for slug in VITAL_SLUGS]
    values = df[value_cols].reset_index(drop=True).copy()
    for slug, normalizer in VITAL_UNIT_NORMALIZERS.items():
        vcol, ucol = f"{slug}_value", f"{slug}_unit"
        values[vcol] = [
            _normalize_vital_value(v, u, normalizer)
            for v, u in zip(df[vcol], df[ucol])
        ]
    imputer = SimpleImputer(strategy="mean")
    imputed = imputer.fit_transform(values)
    pieces.append(pd.DataFrame(imputed, columns=value_cols).round(1))

    # --- date-unknown flags: bool -> 0/1 -----------------------------------
    du_cols = [f"{slug}_date_unknown" for slug in VITAL_SLUGS]
    du = df[du_cols].reset_index(drop=True)
    du = du.apply(lambda col: col.map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int))
    pieces.append(du)

    encoded = pd.concat(pieces, axis=1)

    feature_cols = [c for c in encoded.columns if c != "patient_id"]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(encoded[c])]
    summary = {
        "n_rows": len(encoded),
        "raw_columns": len(df.columns),
        "encoded_feature_columns": len(feature_cols),
        "non_numeric_feature_columns": non_numeric,
        "missing_cells_after": int(encoded[feature_cols].isna().sum().sum()),
    }
    return encoded, summary


def main():
    parser = argparse.ArgumentParser(description="Encode the flattened feature table into a scikit-learn-ready matrix.")
    parser.add_argument("--in", dest="in_path", type=Path, default=DATA_DIR / "patient_features.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "patient_features_encoded.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    encoded, summary = encode(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(args.out, index=False)

    print(f"Wrote {summary['n_rows']} rows x {summary['encoded_feature_columns']} feature columns to {args.out}")
    print(f"  raw columns: {summary['raw_columns']} -> encoded feature columns: {summary['encoded_feature_columns']}")
    print(f"  non-numeric feature columns: {summary['non_numeric_feature_columns'] or 'none (all numeric)'}")
    print(f"  missing cells after encoding: {summary['missing_cells_after']}")


if __name__ == "__main__":
    main()
