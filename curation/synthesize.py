"""Stage C — synthesis of new records for zero-represented diagnosis categories.

Takes `data/curated/rebalanced.jsonl` (Stage C's rebalance output) and fills
the one gap `rebalance.py` documented but couldn't fix: a diagnosis category
with *zero* records can't be rebalanced by duplication, because duplication
can only amplify an existing record, never manufacture one from nothing
(see `rebalance.py`'s docstring). This module is that fix — it generates
genuinely new records via the Anthropic API for whichever categories are
still at zero, up to the same target `rebalance.py` used
(`max(original per-category counts)`, computed from `data/curated/
redacted.jsonl` — the counts *before* rebalance's duplication inflated some
categories past that target).

This is Stage C's final sub-step; its output (`data/curated/
synthesized.jsonl`) is what Stage D consumes.

**Why LLM generation instead of a hand-written template record.** A
hand-written template (fixed diagnosis + one plausible medication + a
couple of vitals) would be faster and free, and for a single missing
category it's tempting to just write one. Rejected because: (1) it doesn't
scale — every zero-count category found would need its own hand-authored
record, which is exactly the kind of manual, non-reusable work this
pipeline is meant to demonstrate automating; (2) an LLM can reason about
*clinically plausible comorbidities* (what else would realistically co-occur
with this diagnosis) and *appropriate medications* in a way a static
template can't without effectively re-deriving a small clinical knowledge
base by hand. The trade-off is cost and non-determinism at the *content*
level (API calls, LLM judgment) — mitigated the same way `extractor.py`
mitigates it: cache-first by default, so repeated dev runs don't re-hit the
API.

**Division of labor: LLM picks *which*, code fills in *how*.** The model
chooses which comorbid diagnoses, which medications, and which vitals (with
plausible values) belong in the record — genuine clinical judgment. But the
model does **not** invent unit strings or dosage text: `diagnoses`,
`medications`, and `vitals` names are constrained to enums drawn directly
from `generate_fhir.py`'s own `CONDITIONS`/`MEDICATIONS`/`OBSERVATIONS`
tables (the same source-of-truth tables `normalize.py` maps *toward*), and
once the model picks a medication or vital name, this module fills in its
canonical dosage text / unit from those tables directly in code. This
guarantees every synthesized record lands in the dataset already in
canonical form — recognizable by `normalize.py`'s lookup tables with zero
translation needed — rather than trusting the model to reproduce exact
canonical strings verbatim.

`patient_id`, `date_of_birth`, and `note_date` are also generated in code,
not by the model: there's no clinical judgment involved in picking a
plausible birth date, and keeping them out of the API call keeps the prompt
(and the cache key) focused on the part that actually needs LLM reasoning.
Both dates are derived from a per-record seeded RNG (same reproducible-hash
approach as `redact.py`'s date shifting) so reruns are deterministic; the
birth-date and note-date ranges mirror `generate_fhir.py`'s own
`make_patient`/`make_condition` ranges for consistency with the rest of the
synthetic dataset.

Every synthesized record is tagged `"synthesized": true` and
`"synthesized_category": "<category>"` — same spirit as `rebalance.py`'s
`rebalance_duplicate_of` tag — so these records stay traceable and
distinguishable from genuinely-extracted ones (e.g. if Stage D ever wants
to report how much of the final dataset is LLM-synthesized rather than
extracted-from-notes).

**Scope: only fills actual gaps, nothing speculative.** This module never
generates more than `target - current_count` records for a category, and
only for categories that are actually at zero after rebalance — it does not
"round out" every category to a nicer number or pad the dataset generally.

**Known side effect, same shape as `rebalance.py`'s: comorbidities cause
overshoot.** On the dev sample, the LLM chose `Essential (primary)
hypertension` and `Obesity, unspecified` as comorbid diagnoses for two of
the three `Hyperlipidemia, unspecified` records it generated — clinically
realistic (all three cluster together in real metabolic syndrome), but it
means those two already-satisfied categories went from 3 to 5 as a
side effect. This is the same overshoot trade-off `rebalance.py` documents
for multi-diagnosis records, just arriving via LLM comorbidity choice
instead of duplication; not fixed here for the same reason — constraining
the model to avoid ever touching an already-satisfied category would mean
either forcing clinically unrealistic single-diagnosis records or a more
complex constrained-generation setup neither justified at this scale.
"""

import argparse
import collections
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.generate_fhir import CONDITIONS, MEDICATIONS, OBSERVATIONS  # noqa: E402
from curation.rebalance import CANONICAL_CONDITION_ORDER, category_counts  # noqa: E402

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REDACTED_PATH = DATA_DIR / "curated" / "redacted.jsonl"
REBALANCED_PATH = DATA_DIR / "curated" / "rebalanced.jsonl"
SYNTHESIZED_PATH = DATA_DIR / "curated" / "synthesized.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "synthesis_cache.json"

MODEL = "claude-haiku-4-5"

MEDICATION_NAMES = [display for _, display, _, _ in MEDICATIONS]
MED_DOSAGE = {display: full_text for _, display, _, full_text in MEDICATIONS}
OBSERVATION_SPECS = {spec["display"]: spec for spec in OBSERVATIONS}
VITAL_NAMES = list(OBSERVATION_SPECS)

# Mirrors generate_fhir.py's make_patient/make_condition date ranges, so
# synthesized records fall in the same plausible window as generated ones.
BIRTH_DATE_RANGE = (date(1940, 1, 1), date(2005, 12, 31))
VISIT_DATE_FLOOR = date(2020, 1, 1)
VISIT_DATE_CEILING = date(2026, 7, 8)

_VITAL_RANGE_HINTS = ", ".join(
    f"{spec['display']} ({spec['low']}-{spec['high']} {spec['unit']})" for spec in OBSERVATIONS
)

SYNTHESIS_TOOL = {
    "name": "record_synthetic_patient",
    "description": (
        "Record a clinically plausible synthetic patient record for a patient "
        "whose primary diagnosis is the given category. Choose realistic "
        "comorbid diagnoses (if any), medications appropriate for treating the "
        "primary diagnosis and any comorbidities, and vitals with plausible "
        "values for this patient."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnoses": {
                "type": "array",
                "description": "1-3 diagnosis names, the primary diagnosis first.",
                "items": {"type": "string", "enum": CANONICAL_CONDITION_ORDER},
            },
            "medications": {
                "type": "array",
                "description": "1-3 medications appropriate for the diagnoses above.",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": MEDICATION_NAMES}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            "vitals": {
                "type": "array",
                "description": f"2-4 vitals with plausible numeric values. Typical ranges: {_VITAL_RANGE_HINTS}.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": VITAL_NAMES},
                        "value": {"type": "number"},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["diagnoses", "medications", "vitals"],
        "additionalProperties": False,
    },
    "strict": True,
}


def load_cache(cache_path):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def generate_fields(client, category):
    """Call the Anthropic API to generate one synthetic record for `category`."""
    prompt = (
        f"Generate one synthetic patient record whose primary diagnosis is "
        f"'{category}'. This is for a synthetic, non-real clinical dataset."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[SYNTHESIS_TOOL],
        tool_choice={"type": "tool", "name": "record_synthetic_patient"},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input


def _seeded_rng(category, index, salt):
    seed = int(hashlib.sha256(f"synthesize:{category}:{index}:{salt}".encode("utf-8")).hexdigest(), 16)
    return random.Random(seed)


def _random_date(rng, start, end):
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta_days, 0)))


def build_record(category, index, generated_fields):
    """Assemble a full curated-schema record from the LLM's category/med/vital
    choices plus code-generated id/dates and code-filled canonical dosage/unit.
    """
    diagnoses = list(dict.fromkeys(generated_fields["diagnoses"]))  # dedupe, preserve order
    if category not in diagnoses:
        diagnoses.insert(0, category)

    medications = [
        {"name": m["name"], "dosage": MED_DOSAGE[m["name"]]}
        for m in generated_fields["medications"]
    ]

    vitals = [
        {"name": v["name"], "value": round(float(v["value"]), 1), "unit": OBSERVATION_SPECS[v["name"]]["unit"]}
        for v in generated_fields["vitals"]
    ]

    birth_rng = _seeded_rng(category, index, "dob")
    birth_date = _random_date(birth_rng, *BIRTH_DATE_RANGE)
    visit_rng = _seeded_rng(category, index, "visit")
    note_date = _random_date(visit_rng, max(birth_date, VISIT_DATE_FLOOR), VISIT_DATE_CEILING)

    id_rng = _seeded_rng(category, index, "id")
    patient_id = f"synth-{id_rng.getrandbits(48):012x}"

    return {
        "patient_id": patient_id,
        "note_date": note_date.isoformat(),
        "date_of_birth": birth_date.isoformat(),
        "diagnoses": [{"name": d} for d in diagnoses],
        "medications": medications,
        "vitals": vitals,
        "synthesized": True,
        "synthesized_category": category,
    }


def read_records(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def synthesize_records(client, records, cache, refresh=False):
    """Return (augmented_records, new_records: list of synthesized records)."""
    original_counts = category_counts(list(read_records(REDACTED_PATH)))
    target = max(original_counts.values()) if original_counts else 0
    current_counts = category_counts(records)

    augmented = list(records)
    new_records = []

    for category in CANONICAL_CONDITION_ORDER:
        deficit = target - current_counts.get(category, 0)
        for i in range(1, deficit + 1):
            cache_key = category  # one cached record per category is enough for this dev-scale demo
            if i > 1:
                cache_key = f"{category}#{i}"
            if refresh or cache_key not in cache:
                cache[cache_key] = generate_fields(client, category)
            record = build_record(category, i, cache[cache_key])
            augmented.append(record)
            new_records.append(record)

    return augmented, new_records, target, original_counts


def main():
    parser = argparse.ArgumentParser(description="Synthesize records to fill zero-represented diagnosis categories.")
    parser.add_argument("--in", dest="in_path", type=Path, default=REBALANCED_PATH)
    parser.add_argument("--out", type=Path, default=SYNTHESIZED_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--refresh", action="store_true", help="Re-call the API even for cached categories")
    args = parser.parse_args()

    records = list(read_records(args.in_path))
    before_counts = category_counts(records)

    client = anthropic.Anthropic()
    cache = load_cache(args.cache)
    augmented, new_records, target, original_counts = synthesize_records(client, records, cache, refresh=args.refresh)
    save_cache(args.cache, cache)

    after_counts = category_counts(augmented)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in augmented:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(augmented)} records ({len(records)} + {len(new_records)} synthesized) to {args.out}")
    print(f"Target per category (from pre-rebalance counts in {REDACTED_PATH.name}): {target}")
    print(f"\n{'diagnosis category':55} {'before':>7} {'after':>7}")
    for category in CANONICAL_CONDITION_ORDER:
        b = before_counts.get(category, 0)
        a = after_counts.get(category, 0)
        flag = "  <- synthesized" if category in {r["synthesized_category"] for r in new_records} else ""
        print(f"{category:55} {b:>7} {a:>7}{flag}")

    if new_records:
        print("\nsynthesized records:")
        for r in new_records:
            meds = [m["name"] for m in r["medications"]]
            print(f"  {r['patient_id']} ({r['synthesized_category']}): dx={[d['name'] for d in r['diagnoses']]} meds={meds}")
    else:
        print("\nNo zero-represented categories found — nothing to synthesize.")


if __name__ == "__main__":
    main()
