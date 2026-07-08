"""Shared parsing helper for reading synthetic FHIR bundles (Stage A output).

Both flatten.py (tabular view) and generate_notes.py (free-text view) need
to walk a Bundle's entries and group them by resource type; this is the one
place that grouping logic lives so the two don't drift out of sync.
"""


def group_bundle_entries(bundle):
    """Split a Bundle's entries into (patient, conditions, medications, observations).

    conditions/medications are lists of resource-derived dicts; observations
    is a dict of {display: [reading, ...]}.
    """
    patient = None
    conditions = []
    medications = []
    observations = {}

    for entry in bundle["entry"]:
        resource = entry["resource"]
        rtype = resource["resourceType"]

        if rtype == "Patient":
            patient = resource
        elif rtype == "Condition":
            conditions.append({
                "code": resource["code"]["coding"][0]["code"],
                "display": resource["code"]["text"],
                "onset": resource.get("onsetDateTime"),
            })
        elif rtype == "MedicationStatement":
            medications.append({
                "code": resource["medicationCodeableConcept"]["coding"][0]["code"],
                "display": resource["medicationCodeableConcept"]["text"],
                "dosage_text": resource.get("dosage", [{}])[0].get("text", ""),
                "effective": resource.get("effectiveDateTime"),
            })
        elif rtype == "Observation":
            display = resource["code"]["text"]
            observations.setdefault(display, []).append({
                "date": resource.get("effectiveDateTime"),
                "value": resource["valueQuantity"]["value"],
                "unit": resource["valueQuantity"]["unit"],
            })

    if patient is None:
        raise ValueError("Bundle has no Patient resource")

    return patient, conditions, medications, observations
