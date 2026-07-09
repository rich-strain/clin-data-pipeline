"""Synthetic FHIR resource generation for Stage A.

Produces one Bundle per synthetic patient containing a Patient resource plus
a handful of Condition, Observation, and MedicationStatement resources. This
is the system-of-record form for the pipeline — everything downstream
(flatten.py's tabular view, and eventually the training data) is derived
from these FHIR resources, not the other way around.

Messiness toggle (`messy=True`) introduces realistic EHR-data problems
(inconsistent units, missing optional fields, free-text dosage shorthand)
directly at generation time, so Stage C's curation steps have real problems
to fix.

Patient resources carry a synthetic MRN (`identifier`) and a synthetic
address (`address`) — both flow downstream into note prose and extraction
(see `generate_notes.py`/`extractor.py`) so Stage C's redaction step has a
real field to act on for these two Safe Harbor identifiers, matching
name/DOB. See CLAUDE.md Resolved decisions #9.
"""

import argparse
import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"

FIRST_NAMES_MALE = [
    "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Paul",
    "Steven",
]
FIRST_NAMES_FEMALE = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret", "Betty",
    "Sandra",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

STREET_NAMES = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Elm St", "Pine Rd",
    "Washington Ave", "Park Blvd", "Sunset Dr", "Lake St",
]
# (city, state) — plausible-shaped, not real address-to-patient mappings.
CITIES_STATES = [
    ("Springfield", "IL"), ("Franklin", "TX"), ("Greenville", "SC"),
    ("Clinton", "OH"), ("Salem", "OR"), ("Georgetown", "KY"),
    ("Arlington", "VA"), ("Madison", "WI"), ("Bristol", "CT"),
    ("Fairview", "NC"),
]

# (ICD-10-CM code, display)
CONDITIONS = [
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("J45.909", "Unspecified asthma, uncomplicated"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
    ("G43.909", "Migraine, unspecified, not intractable"),
    ("E66.9", "Obesity, unspecified"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
    ("E03.9", "Hypothyroidism, unspecified"),
    ("F32.9", "Major depressive disorder, single episode, unspecified"),
]
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"

# (RxNorm code, display, dosage shorthand, dosage full text)
MEDICATIONS = [
    ("6809", "Metformin 500 MG Oral Tablet", "500mg PO BID", "Take 500 mg by mouth twice daily"),
    ("29046", "Lisinopril 10 MG Oral Tablet", "10mg PO QD", "Take 10 mg by mouth once daily"),
    ("83367", "Atorvastatin 20 MG Oral Tablet", "20mg PO QHS", "Take 20 mg by mouth at bedtime"),
    ("435", "Albuterol 90 MCG Inhaler", "2puff q4h PRN", "Inhale 2 puffs every 4 hours as needed"),
    ("966", "Levothyroxine 75 MCG Oral Tablet", "75mcg PO QAM", "Take 75 mcg by mouth every morning"),
    ("36437", "Sertraline 50 MG Oral Tablet", "50mg PO QD", "Take 50 mg by mouth once daily"),
    ("7646", "Omeprazole 20 MG Oral Capsule", "20mg PO QD", "Take 20 mg by mouth once daily"),
    ("17767", "Amlodipine 5 MG Oral Tablet", "5mg PO QD", "Take 5 mg by mouth once daily"),
]
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"


def _c_to_f(c):
    return round(c * 9 / 5 + 32, 1)


def _kg_to_lb(kg):
    return round(kg * 2.20462, 1)


def _cm_to_in(cm):
    return round(cm / 2.54, 1)


# (LOINC code, display, low, high, canonical unit, messy alt unit, converter)
OBSERVATIONS = [
    {"code": "8867-4", "display": "Heart rate", "low": 55, "high": 100,
     "unit": "beats/minute", "alt_unit": None, "convert": None},
    {"code": "8480-6", "display": "Systolic blood pressure", "low": 100, "high": 140,
     "unit": "mm[Hg]", "alt_unit": None, "convert": None},
    {"code": "8310-5", "display": "Body temperature", "low": 36.1, "high": 37.8,
     "unit": "Cel", "alt_unit": "[degF]", "convert": _c_to_f},
    {"code": "29463-7", "display": "Body weight", "low": 55, "high": 110,
     "unit": "kg", "alt_unit": "[lb_av]", "convert": _kg_to_lb},
    {"code": "8302-2", "display": "Body height", "low": 150, "high": 190,
     "unit": "cm", "alt_unit": "[in_i]", "convert": _cm_to_in},
    {"code": "2339-0", "display": "Glucose", "low": 70, "high": 180,
     "unit": "mg/dL", "alt_unit": None, "convert": None},
]
LOINC_SYSTEM = "http://loinc.org"


def _random_date(rng, start, end):
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta_days, 0)))


def _new_id():
    return str(uuid.uuid4())


def _make_address(rng):
    street_number = rng.randint(100, 9999)
    street_name = rng.choice(STREET_NAMES)
    city, state = rng.choice(CITIES_STATES)
    return {
        "line": [f"{street_number} {street_name}"],
        "city": city,
        "state": state,
        "postalCode": f"{rng.randint(10000, 99999)}",
    }


def make_patient(rng, messy):
    gender = rng.choice(["male", "female"])
    given = rng.choice(FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE)
    family = rng.choice(LAST_NAMES)
    birth_date = _random_date(rng, date(1940, 1, 1), date(2005, 12, 31))
    patient_id = _new_id()

    # Address (and its messy-drop checks, and MRN's) are drawn from a
    # patient_id-seeded local RNG, not the shared `rng` stream. patient_id
    # already comes from uuid4 (unseeded by `rng`), so this loses no
    # reproducibility versus before, and it keeps this addition purely
    # additive: it can't shift the shared stream's position and silently
    # reshuffle every other field for this patient — or every later
    # patient's conditions/observations/medications — just from adding an
    # address.
    addr_rng = random.Random(patient_id)

    resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "system": "urn:oid:synthetic-mrn",
            "value": f"MRN{rng.randint(100000, 999999)}",
        }],
        "name": [{"family": family, "given": [given]}],
        "gender": gender,
        "birthDate": birth_date.isoformat(),
        "address": [_make_address(addr_rng)],
    }

    if messy and rng.random() < 0.15:
        del resource["gender"]

    # Same 0.15 messy-drop rate as gender above — MRN and address are both
    # real-world "sometimes missing on intake" fields, not fields with a
    # different plausible drop rate than demographics generally.
    if messy and addr_rng.random() < 0.15:
        del resource["identifier"]

    if messy and addr_rng.random() < 0.15:
        del resource["address"]

    return resource, patient_id, birth_date


def make_condition(rng, patient_id, birth_date, messy):
    code, display = rng.choice(CONDITIONS)
    onset = _random_date(rng, max(birth_date, date(2015, 1, 1)), date(2026, 7, 8))

    resource = {
        "resourceType": "Condition",
        "id": _new_id(),
        "subject": {"reference": f"Patient/{patient_id}"},
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active",
            }],
        },
        "code": {
            "coding": [{"system": ICD10_SYSTEM, "code": code, "display": display}],
            "text": display,
        },
        "onsetDateTime": onset.isoformat(),
    }

    if messy and rng.random() < 0.2:
        del resource["clinicalStatus"]

    return resource


def make_observation(rng, patient_id, birth_date, messy):
    spec = rng.choice(OBSERVATIONS)
    value = round(rng.uniform(spec["low"], spec["high"]), 1)
    unit = spec["unit"]

    if messy and spec["convert"] and rng.random() < 0.4:
        value = spec["convert"](value)
        unit = spec["alt_unit"]

    effective = _random_date(rng, max(birth_date, date(2020, 1, 1)), date(2026, 7, 8))

    resource = {
        "resourceType": "Observation",
        "id": _new_id(),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [{"system": LOINC_SYSTEM, "code": spec["code"], "display": spec["display"]}],
            "text": spec["display"],
        },
        "effectiveDateTime": effective.isoformat(),
        "valueQuantity": {
            "value": value,
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": unit,
        },
    }

    if messy and rng.random() < 0.15:
        del resource["effectiveDateTime"]

    return resource


def make_medication_statement(rng, patient_id, birth_date, messy):
    code, display, shorthand, full_text = rng.choice(MEDICATIONS)
    asserted = _random_date(rng, max(birth_date, date(2020, 1, 1)), date(2026, 7, 8))
    dosage_text = shorthand if (messy and rng.random() < 0.5) else full_text

    resource = {
        "resourceType": "MedicationStatement",
        "id": _new_id(),
        "status": "active",
        "subject": {"reference": f"Patient/{patient_id}"},
        "medicationCodeableConcept": {
            "coding": [{"system": RXNORM_SYSTEM, "code": code, "display": display}],
            "text": display,
        },
        "effectiveDateTime": asserted.isoformat(),
        "dosage": [{"text": dosage_text}],
    }

    if messy and rng.random() < 0.15:
        del resource["dosage"]

    return resource


def generate_patient_bundle(rng, messy):
    patient_resource, patient_id, birth_date = make_patient(rng, messy)
    entries = [{"resource": patient_resource}]

    for _ in range(rng.randint(1, 3)):
        entries.append({"resource": make_condition(rng, patient_id, birth_date, messy)})
    for _ in range(rng.randint(2, 5)):
        entries.append({"resource": make_observation(rng, patient_id, birth_date, messy)})
    for _ in range(rng.randint(1, 3)):
        entries.append({"resource": make_medication_statement(rng, patient_id, birth_date, messy)})

    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def generate_dataset(n_patients, messy=False, seed=42):
    rng = random.Random(seed)
    return [generate_patient_bundle(rng, messy) for _ in range(n_patients)]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FHIR patient bundles.")
    parser.add_argument("--n-patients", type=int, default=25)
    parser.add_argument("--messy", action="store_true", help="Introduce realistic messiness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "fhir_bundles.json")
    args = parser.parse_args()

    bundles = generate_dataset(args.n_patients, messy=args.messy, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundles, indent=2))

    print(f"Wrote {len(bundles)} patient bundles ({'messy' if args.messy else 'clean'}) to {args.out}")


if __name__ == "__main__":
    main()
