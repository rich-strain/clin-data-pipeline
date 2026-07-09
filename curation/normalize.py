"""Stage C — normalization of Stage B's raw extraction output.

Takes `data/extracted/extractions.jsonl` (the "LLM dump": free-text wording
preserved on purpose, per `extraction/extractor.py`) and produces a
consistently-formatted version: canonical diagnosis names, canonical vital
names/units (with values converted, not just relabeled), and canonical
medication dosage text.

**Design decision — lookup tables grounded in the generator's own source of
truth, not free-form NLP.** Stage A's `generation/generate_fhir.py` already
defines the small, closed set of conditions/medications/vitals this synthetic
dataset ever produces (`CONDITIONS`, `MEDICATIONS`, `OBSERVATIONS`), and
`generation/generate_notes.py` defines exactly which abbreviations/shorthand
those get rendered as in note text. Since extraction pulls from that same
closed vocabulary, normalization here is a lookup against those known
variants rather than a general medical-abbreviation parser or an
NLP/fuzzy-matching pass. Trade-off: this only works because the dataset is
synthetic and closed-vocabulary — a real EHR feed would need a real
terminology service (RxNorm/LOINC/SNOMED mapping), which is out of scope for
a portfolio pipeline. Values that don't match a known variant are left as-is
and reported, not silently dropped or guessed at.

`CONDITIONS`/`MEDICATIONS`/`OBSERVATIONS` are imported directly from
`generation.generate_fhir` (a clean import, single source of truth, no
duplication). `generate_notes.py`'s abbreviation/label tables are *not*
imported — that module does a script-relative `from fhir_common import ...`
that only resolves when `generation/` itself is on `sys.path`, not just the
repo root — so the small, stable abbreviation tables are duplicated here
directly instead of adding import-path hacks to reach them.

**Performance note:** all pattern matching here is against a handful of
compiled regexes and small fixed dicts, applied once per record — no
per-record recompilation, no repeated full-text scanning.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.generate_fhir import CONDITIONS, MEDICATIONS, OBSERVATIONS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACTED_PATH = DATA_DIR / "extracted" / "extractions.jsonl"
NORMALIZED_PATH = DATA_DIR / "curated" / "normalized.jsonl"

# --- Diagnoses -------------------------------------------------------------

CANONICAL_CONDITIONS = [display for _, display in CONDITIONS]
# Longest first, so a startswith-match picks the most specific canonical
# string when one is a prefix of another (none currently are, but this keeps
# the match order defensible as the condition list grows).
_CONDITIONS_BY_LENGTH = sorted(CANONICAL_CONDITIONS, key=len, reverse=True)

# Mirrors generation/generate_notes.py's CONDITION_ABBREVIATIONS (see module
# docstring for why it's duplicated rather than imported).
CONDITION_ABBREVIATIONS = {
    "Type 2 diabetes mellitus without complications": "T2DM",
    "Essential (primary) hypertension": "HTN",
    "Unspecified asthma, uncomplicated": "asthma",
    "Hyperlipidemia, unspecified": "HLD",
    "Chronic obstructive pulmonary disease, unspecified": "COPD",
    "Migraine, unspecified, not intractable": "migraine",
    "Obesity, unspecified": "obesity",
    "Gastro-esophageal reflux disease without esophagitis": "GERD",
    "Hypothyroidism, unspecified": "hypothyroidism",
    "Major depressive disorder, single episode, unspecified": "MDD",
}
_ABBREV_TO_CANONICAL = {v.lower(): k for k, v in CONDITION_ABBREVIATIONS.items()}

# Notes render PMH entries as "{condition} (dx {onset date})" (see
# generate_notes.py's build_note_text); extraction sometimes captures that
# trailing date parenthetical as part of the diagnosis name. Stripping it
# here is a normalization/consistency fix (the name field should just be the
# name); the *date itself* is a visit-adjacent field Stage C's redaction step
# should also account for. Flagged in docs/design_decisions.md as an
# upstream fix candidate: extraction could capture onset date as its own
# field instead of leaving it embedded in free text.
_TRAILING_DX_DATE_RE = re.compile(r"\s*\(dx\s+[^)]*\)\s*$", re.IGNORECASE)


def normalize_diagnosis_name(raw_name):
    """Return (canonical_name, matched: bool)."""
    stripped = _TRAILING_DX_DATE_RE.sub("", raw_name).strip()

    abbrev_match = _ABBREV_TO_CANONICAL.get(stripped.lower())
    if abbrev_match:
        return abbrev_match, True

    for canonical in _CONDITIONS_BY_LENGTH:
        if stripped.lower() == canonical.lower():
            return canonical, True
        # Handles LLM extraction artifacts like "<canonical> management"
        # (bled in from a "Continue management of {condition}" sentence).
        if stripped.lower().startswith(canonical.lower()):
            return canonical, True

    return stripped, False


# --- Vitals ------------------------------------------------------------

# Aliases observed in note text (generate_notes.py's VITALS_LABELS,
# full/short forms) plus paraphrase variants observed in the actual
# extraction sample (e.g. the model returning "Heart Rate" / "Blood
# Pressure" instead of the note's exact "Heart rate" / "Blood pressure
# (systolic)"). Matching is case-insensitive.
_VITAL_ALIASES = {
    "heart rate": "Heart rate", "hr": "Heart rate",
    "systolic blood pressure": "Systolic blood pressure",
    "blood pressure (systolic)": "Systolic blood pressure",
    "blood pressure": "Systolic blood pressure", "bp": "Systolic blood pressure",
    "body temperature": "Body temperature",
    "temperature": "Body temperature", "temp": "Body temperature",
    "body weight": "Body weight", "weight": "Body weight", "wt": "Body weight",
    "body height": "Body height", "height": "Body height", "ht": "Body height",
    "glucose": "Glucose", "glu": "Glucose",
}

_OBSERVATION_SPECS = {spec["display"]: spec for spec in OBSERVATIONS}

# Inverse of generate_fhir.py's forward converters (canonical -> alt unit),
# needed here because messy *input* may already be in the alt unit.
_INVERSE_CONVERTERS = {
    "Body temperature": lambda f: round((f - 32) * 5 / 9, 1),   # degF -> Cel
    "Body weight": lambda lb: round(lb / 2.20462, 1),           # lb -> kg
    "Body height": lambda inch: round(inch * 2.54, 1),          # in -> cm
}

# Whether a missing unit's magnitude implies the alt unit, based on the
# canonical low/high range each spec already declares — e.g. body weight's
# canonical (kg) range tops out at 110, so a value above that is almost
# certainly already in pounds.
_ALT_UNIT_IF = {
    "Body temperature": lambda v, spec: v > spec["high"],
    "Body weight": lambda v, spec: v > spec["high"],
    "Body height": lambda v, spec: v < spec["low"],
}


def normalize_vital(raw_name, raw_value, unit):
    """Return (canonical_name, canonical_value, canonical_unit, matched: bool).

    `raw_value` arrives as a string (the extraction tool schema captures
    vitals as free-text-preserving strings); normalizing to a numeric type
    is itself part of consistent formatting, so the returned value is a
    float.
    """
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return raw_name, raw_value, unit, False

    canonical_name = _VITAL_ALIASES.get(raw_name.strip().lower())
    if canonical_name is None:
        return raw_name, value, unit, False

    spec = _OBSERVATION_SPECS[canonical_name]
    canonical_unit = spec["unit"]
    alt_unit = spec["alt_unit"]

    if unit is None:
        alt_check = _ALT_UNIT_IF.get(canonical_name)
        if alt_check is not None and alt_check(value, spec):
            return canonical_name, _INVERSE_CONVERTERS[canonical_name](value), canonical_unit, True
        return canonical_name, value, canonical_unit, True

    if unit == canonical_unit:
        return canonical_name, value, canonical_unit, True

    if alt_unit is not None and unit == alt_unit:
        return canonical_name, _INVERSE_CONVERTERS[canonical_name](value), canonical_unit, True

    # Unrecognized unit string: leave value/unit untouched, flag as unmatched.
    return canonical_name, value, unit, False


# --- Medications ---------------------------------------------------------

# {medication display: {known dosage text variant: canonical (full-text) dosage}}
_DOSAGE_LOOKUP = {
    display: {shorthand: full_text, full_text: full_text}
    for _, display, shorthand, full_text in MEDICATIONS
}
_MISSING_DOSAGE_MARKERS = {"", "dosage not recorded", "none"}


def normalize_dosage(med_name, dosage_text):
    """Return (canonical_dosage_or_None, matched: bool)."""
    if dosage_text is None or dosage_text.strip().lower() in _MISSING_DOSAGE_MARKERS:
        return None, True

    variants = _DOSAGE_LOOKUP.get(med_name)
    if variants and dosage_text in variants:
        return variants[dosage_text], True

    return dosage_text, False


# --- Record-level normalization -------------------------------------------

def normalize_record(record):
    """Normalize one extraction record. Returns (normalized_record, unmatched_notes)."""
    unmatched = []

    diagnoses = []
    seen_diagnoses = set()
    for dx in record["diagnoses"]:
        name, matched = normalize_diagnosis_name(dx["name"])
        if not matched:
            unmatched.append(f"diagnosis: {dx['name']!r} -> {name!r}")
        # Note generation re-rolls each condition's abbreviation independently
        # per section (CC/HPI/PMH/Plan), so the same underlying diagnosis can
        # arrive as multiple differently-worded mentions; normalizing to a
        # canonical name is what surfaces them as duplicates, so dedup here
        # rather than carrying "mentioned N times" forward as "N diagnoses".
        if name in seen_diagnoses:
            continue
        seen_diagnoses.add(name)
        diagnoses.append({"name": name})

    medications = []
    for med in record["medications"]:
        dosage, matched = normalize_dosage(med["name"], med["dosage"])
        medications.append({"name": med["name"], "dosage": dosage})
        if not matched:
            unmatched.append(f"dosage for {med['name']!r}: {med['dosage']!r}")

    vitals = []
    for v in record["vitals"]:
        name, value, unit, matched = normalize_vital(v["name"], v["value"], v["unit"])
        vitals.append({"name": name, "value": value, "unit": unit})
        if not matched:
            unmatched.append(f"vital: {v['name']!r} ({v['value']!r} {v['unit']!r})")

    normalized = {**record, "diagnoses": diagnoses, "medications": medications, "vitals": vitals}
    return normalized, unmatched


def read_extractions(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Normalize Stage B extraction output.")
    parser.add_argument("--in", dest="in_path", type=Path, default=EXTRACTED_PATH)
    parser.add_argument("--out", type=Path, default=NORMALIZED_PATH)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    all_unmatched = []
    with args.out.open("w") as f:
        for record in read_extractions(args.in_path):
            normalized, unmatched = normalize_record(record)
            f.write(json.dumps(normalized) + "\n")
            count += 1
            all_unmatched.extend(unmatched)

    print(f"Wrote {count} normalized records to {args.out}")
    if all_unmatched:
        print(f"{len(all_unmatched)} values left unmatched (not in the known vocabulary):")
        for item in all_unmatched:
            print(f"  - {item}")
    else:
        print("All values matched a known canonical form.")


if __name__ == "__main__":
    main()
