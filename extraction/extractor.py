"""Stage B — LLM-based structured extraction from synthetic clinical notes.

Pulls diagnosis/medication/dosage/vitals fields (plus the note's PHI-bearing
header fields — patient name, DOB, MRN, address — since Stage C's redaction
step needs a concrete structured field to strip for each) out of each
free-text note via the Anthropic API. This is the raw "LLM dump" —
normalization, redaction, rebalancing, and synthesis all happen downstream
in Stage C.

`mrn` and `address` are nullable (unlike `patient_name`/`date_of_birth`,
which the note generator never omits): messy generation can drop either
field from the Patient resource entirely, so the note header sometimes
doesn't have one to extract — same nullability pattern already used below
for `dosage`/vital `unit`.

**Performance note (bulk text processing):** each note is sent to the model
as a single API call — there's no local regex/parsing pass over note text at
all, let alone a repeated one, so the usual "avoid naive repeated regex over
large files" concern doesn't apply here. The one thing that matters at scale
is not loading the whole notes file into memory: notes are read and written
one JSONL line at a time (`read_notes` is a generator), so this scales to a
notes file far larger than fits in memory. The real cost driver is API calls,
which is why caching (below) is the priority, not parsing speed.

**Caching.** Extraction is cache-first by default, keyed on a SHA-256 hash of
the note text. The cache (`extraction/cache/extraction_cache.json`) is a
small, committed dev seed of real API responses (cheap on Haiku) — Stage C
work iterates against that cache instead of re-hitting the API on every dev
cycle. Pass --refresh to force live calls and overwrite cached entries.
"""

import argparse
import hashlib
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NOTES_PATH = DATA_DIR / "generated" / "clinical_notes.jsonl"
EXTRACTED_PATH = DATA_DIR / "extracted" / "extractions.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "extraction_cache.json"

MODEL = "claude-haiku-4-5"

EXTRACTION_TOOL = {
    "name": "record_clinical_extraction",
    "description": (
        "Record the structured fields extracted from a clinical note: the "
        "patient identifying header, diagnoses, medications, and vitals. "
        "Preserve the note's own wording (including abbreviations, missing "
        "units, or shorthand) rather than normalizing it — normalization "
        "happens in a later curation step."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "patient_name": {
                "type": "string",
                "description": "Patient name as it appears in the note header.",
            },
            "date_of_birth": {
                "type": "string",
                "description": "Patient date of birth as it appears in the note header.",
            },
            "mrn": {
                "type": ["string", "null"],
                "description": "Patient MRN as it appears in the note header, or null if not present.",
            },
            "address": {
                "type": ["string", "null"],
                "description": "Patient address as it appears in the note header, or null if not present.",
            },
            "diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dosage": {"type": ["string", "null"]},
                    },
                    "required": ["name", "dosage"],
                    "additionalProperties": False,
                },
            },
            "vitals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "string"},
                        "unit": {"type": ["string", "null"]},
                    },
                    "required": ["name", "value", "unit"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["patient_name", "date_of_birth", "mrn", "address", "diagnoses", "medications", "vitals"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _note_hash(note_text):
    return hashlib.sha256(note_text.encode("utf-8")).hexdigest()


def load_cache(cache_path):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def extract_fields(client, note_text):
    """Call the Anthropic API to extract structured fields from one note."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_clinical_extraction"},
        messages=[{"role": "user", "content": note_text}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input


def read_notes(notes_path):
    with notes_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_notes(notes, client, cache, refresh=False):
    for note in notes:
        note_text = note["note_text"]
        key = _note_hash(note_text)
        if refresh or key not in cache:
            cache[key] = extract_fields(client, note_text)
        yield {
            "patient_id": note["patient_id"],
            "note_date": note["note_date"],
            **cache[key],
        }


def main():
    parser = argparse.ArgumentParser(description="Extract structured fields from synthetic clinical notes.")
    parser.add_argument("--in", dest="in_path", type=Path, default=NOTES_PATH)
    parser.add_argument("--out", type=Path, default=EXTRACTED_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N notes")
    parser.add_argument("--refresh", action="store_true", help="Re-call the API even for cached notes")
    args = parser.parse_args()

    notes = read_notes(args.in_path)
    if args.limit is not None:
        notes = (note for i, note in enumerate(notes) if i < args.limit)

    client = anthropic.Anthropic()
    cache = load_cache(args.cache)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w") as f:
        for extraction in extract_notes(notes, client, cache, refresh=args.refresh):
            f.write(json.dumps(extraction) + "\n")
            count += 1

    save_cache(args.cache, cache)
    print(f"Wrote {count} extractions to {args.out} (cache: {len(cache)} entries at {args.cache})")


if __name__ == "__main__":
    main()
