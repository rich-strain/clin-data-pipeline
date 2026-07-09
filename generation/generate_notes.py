"""Generates synthetic free-text clinical notes from Stage A's FHIR bundles.

Notes are grounded in each patient's actual conditions/medications/vitals
(as read from the Bundle) rather than generated from independent random
facts. Two reasons:

1. Realism — a real clinical note narrates a real patient's actual history,
   it doesn't invent unrelated facts.
2. It gives Stage B (LLM extraction) a ground truth to be checked against:
   since the notes are built from known structured facts, extracted fields
   can later be compared back to the source Condition/MedicationStatement/
   Observation resources to sanity-check extraction quality.

Notes are assembled from sentence templates (Mad-libs style), not an LLM
call — that's deliberately reserved for Stage B. The `--messy` toggle
simulates how real clinicians actually write: clinical abbreviations
(HTN, T2DM, GERD, MDD...), terse section headers, dropped sections, and
vitals reported without units — on top of whatever messiness Stage A's FHIR
generation already introduced (e.g. dosage shorthand carried straight
through from `dosage.text`).

Abbreviation choice is re-rolled per mention rather than fixed per patient,
so the same condition can appear abbreviated in one section and spelled out
in another within a single note — real clinicians are inconsistent about
this, and it's exactly the kind of terminology drift Stage C's normalization
step exists to clean up.

The note header also carries the patient's synthetic MRN and address (when
the Patient resource has them — messy generation can drop either), right
alongside name/DOB, since a real chart header includes them too. See
CLAUDE.md Resolved decisions #9.
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path

from fhir_common import group_bundle_entries

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

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

VITALS_LABELS = {
    "Heart rate": ("Heart rate", "HR"),
    "Systolic blood pressure": ("Blood pressure (systolic)", "BP"),
    "Body temperature": ("Temperature", "Temp"),
    "Body weight": ("Weight", "Wt"),
    "Body height": ("Height", "Ht"),
    "Glucose": ("Glucose", "Glu"),
}

SECTION_HEADERS = {
    "cc": ("Chief Complaint", "CC"),
    "hpi": ("History of Present Illness", "HPI"),
    "pmh": ("Past Medical History", "PMH"),
    "meds": ("Current Medications", "Meds"),
    "vitals": ("Vitals", "VS"),
    "plan": ("Assessment/Plan", "A/P"),
}

CC_TEMPLATES = [
    "Follow-up visit for {condition} management.",
    "Presents for routine follow-up of {condition}.",
    "Here today to discuss {condition}.",
]
CC_GENERIC = ["Presents for annual physical exam.", "Here today for a general wellness check."]

HPI_TEMPLATES = [
    "Patient reports doing well on current regimen for {condition}, diagnosed {onset}.",
    "{condition} (dx {onset}) remains stable; no new complaints today.",
    "Continues management of {condition}, first diagnosed {onset}. No acute concerns reported.",
]

PLAN_TEMPLATES = [
    "Continue current management of {condition}.",
    "Continue present treatment for {condition}; recheck at next visit.",
    "No changes to {condition} management at this time.",
]


def _parse_date(value):
    return date.fromisoformat(value) if value else None


def _note_date(conditions, medications, observations):
    all_dates = [c["onset"] for c in conditions if c["onset"]]
    all_dates += [m["effective"] for m in medications if m["effective"]]
    all_dates += [o["date"] for readings in observations.values() for o in readings if o["date"]]
    parsed = [_parse_date(d) for d in all_dates]
    return max(parsed) if parsed else date(2026, 7, 8)


def _latest_reading(readings):
    dated = [r for r in readings if r["date"]]
    if dated:
        return max(dated, key=lambda r: r["date"])
    return readings[-1] if readings else None


def _condition_label(display, messy, rng):
    if messy and display in CONDITION_ABBREVIATIONS and rng.random() < 0.7:
        return CONDITION_ABBREVIATIONS[display]
    return display


def _header(key, messy):
    full, short = SECTION_HEADERS[key]
    return short if messy else full


def _mrn(patient):
    identifiers = patient.get("identifier")
    return identifiers[0]["value"] if identifiers else None


def _address_str(patient):
    addresses = patient.get("address")
    if not addresses:
        return None
    addr = addresses[0]
    street = ", ".join(addr.get("line", []))
    city = addr.get("city", "")
    state = addr.get("state", "")
    postal_code = addr.get("postalCode", "")
    return f"{street}, {city}, {state} {postal_code}".strip()


def build_note_text(patient, conditions, medications, observations, messy, rng):
    name = patient.get("name", [{}])[0]
    full_name = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
    note_date = _note_date(conditions, medications, observations)
    mrn = _mrn(patient)
    address = _address_str(patient)

    lines = []
    lines.append(f"Patient: {full_name}, DOB {patient['birthDate']}"
                  + (f", {patient['gender'].capitalize()}" if patient.get("gender") else "")
                  + (f", MRN {mrn}" if mrn else ""))
    if address:
        lines.append(f"Address: {address}")
    lines.append(f"Visit Date: {note_date.isoformat()}")
    lines.append("")

    primary_condition = conditions[0] if conditions else None
    cc_label = _header("cc", messy)
    if primary_condition:
        template = rng.choice(CC_TEMPLATES)
        cc_text = template.format(condition=_condition_label(primary_condition["display"], messy, rng))
    else:
        cc_text = rng.choice(CC_GENERIC)
    lines.append(f"{cc_label}: {cc_text}")
    lines.append("")

    # HPI section is occasionally dropped in messy notes -- a terse,
    # rushed real-world note skipping narrative in favor of the list sections.
    if conditions and not (messy and rng.random() < 0.2):
        hpi_label = _header("hpi", messy)
        sentences = []
        for c in conditions:
            template = rng.choice(HPI_TEMPLATES)
            onset = c["onset"] or "an unknown date"
            sentences.append(template.format(condition=_condition_label(c["display"], messy, rng), onset=onset))
        lines.append(f"{hpi_label}: {' '.join(sentences)}")
        lines.append("")

    if conditions:
        pmh_label = _header("pmh", messy)
        entries = [f"{_condition_label(c['display'], messy, rng)} (dx {c['onset'] or 'unknown'})" for c in conditions]
        lines.append(f"{pmh_label}: {', '.join(entries)}.")
        lines.append("")

    if medications:
        meds_label = _header("meds", messy)
        entries = [f"{m['display']} - {m['dosage_text'] or 'dosage not recorded'}" for m in medications]
        lines.append(f"{meds_label}: {'; '.join(entries)}.")
        lines.append("")

    # Vitals are occasionally omitted entirely -- not every encounter note
    # includes a full vitals recheck.
    if observations and not (messy and rng.random() < 0.15):
        vitals_label = _header("vitals", messy)
        parts = []
        for display, readings in observations.items():
            latest = _latest_reading(readings)
            if not latest:
                continue
            full_label, short_label = VITALS_LABELS.get(display, (display, display))
            label = short_label if messy else full_label
            unit = "" if (messy and rng.random() < 0.4) else f" {latest['unit']}"
            parts.append(f"{label} {latest['value']}{unit}")
        lines.append(f"{vitals_label}: {', '.join(parts)}.")
        lines.append("")

    if conditions:
        plan_label = _header("plan", messy)
        plan_sentences = [
            rng.choice(PLAN_TEMPLATES).format(condition=_condition_label(c["display"], messy, rng))
            for c in conditions
        ]
        lines.append(f"{plan_label}: {' '.join(plan_sentences)}")

    return "\n".join(lines).strip()


def generate_note(bundle, messy, rng):
    patient, conditions, medications, observations = group_bundle_entries(bundle)
    note_text = build_note_text(patient, conditions, medications, observations, messy, rng)
    return {
        "patient_id": patient["id"],
        "note_date": _note_date(conditions, medications, observations).isoformat(),
        "note_text": note_text,
    }


def generate_notes(bundles, messy=False, seed=42):
    rng = random.Random(seed)
    return [generate_note(bundle, messy, rng) for bundle in bundles]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical notes from FHIR bundles.")
    parser.add_argument("--in", dest="in_path", type=Path, default=DATA_DIR / "fhir_bundles.json")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "clinical_notes.jsonl")
    parser.add_argument("--messy", action="store_true", help="Introduce realistic note-writing messiness")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundles = json.loads(args.in_path.read_text())
    notes = generate_notes(bundles, messy=args.messy, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for note in notes:
            f.write(json.dumps(note) + "\n")

    print(f"Wrote {len(notes)} notes ({'messy' if args.messy else 'clean'}) to {args.out}")


if __name__ == "__main__":
    main()
