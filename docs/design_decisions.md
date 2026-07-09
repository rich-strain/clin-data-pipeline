# Design Decisions

Rationale log for choices made in this pipeline, kept honest and specific —
what was actually built, and why that approach over the alternatives.

## Stage A/flatten — missing `effectiveDateTime` on Observations

**The problem:** `generate_fhir.py`'s messiness toggle can drop an
Observation's `effectiveDateTime` (realistic — real EHR feeds do have
readings with no reliable timestamp). `flatten.py` picks "most recent
reading" per vital by that date. When every reading for a vital lacks a
date, "most recent" is unknowable.

**Decision: fix in `flatten.py`, not the generator.** Dropping the timestamp
is legitimate EHR messiness worth keeping — it's exactly the kind of problem
a curation stage should have to deal with, not something to sanitize away at
generation time. The fix instead makes the ambiguity explicit: when no
reading has a date, `flatten.py` still emits a value (falling back to
generation order, the best available signal) but adds a companion
`{vital}_date_unknown` boolean column. Downstream consumers can see plainly
when "most recent" isn't a reliable claim, rather than the table silently
implying every value is properly dated.

**Alternative considered:** drop the value entirely (`None`) when the date
is unknown, rather than keeping a best-guess value with a flag. Rejected —
it throws away a reading that's still probably informative (generation
order is not random) for one that's more conservative but loses signal for
no measurable benefit in this synthetic-only pipeline.

## Stage C — normalize

**The problem:** Stage B's extraction output preserves the note's own
wording on purpose (see `extractor.py`'s docstring) — clinical
abbreviations (`HTN`, `T2DM`, `GERD`...), missing/inconsistent vital units,
and dosage shorthand (`20mg PO QD` vs. `Take 20 mg by mouth once daily`) all
pass through untouched.

**Decision: lookup tables grounded in the generator's own source-of-truth
tables (`CONDITIONS`, `MEDICATIONS`, `OBSERVATIONS` in `generate_fhir.py`),
not a general NLP/regex abbreviation parser.** Since this is a synthetic,
closed-vocabulary dataset — every diagnosis, medication, and vital the
notes ever mention comes from those same fixed lists — a lookup table is
exact where a fuzzy/NLP approach would only be approximately right, and it's
transparent enough to defend line by line. The trade-off is that this only
works *because* the dataset is closed-vocabulary; a real EHR feed would need
an actual terminology service (RxNorm/LOINC/SNOMED), which is out of scope
here. Anything that doesn't match a known variant is left unchanged and
reported (`normalize.py`'s `--in`/`--out` run prints an unmatched-values
list), not silently dropped or guessed at — on this dataset, everything
matched.

**Vital unit imputation heuristic:** when a vital's unit is missing (an
`OBSERVATIONS`-alt-unit vital — weight, height, temperature — can arrive in
either canonical or alt units with no unit string), the missing unit is
inferred from the value's magnitude relative to `OBSERVATIONS`'s own
declared `low`/`high` range for that vital (e.g. canonical body weight tops
out at 110 kg, so a value above that is almost certainly already in
pounds). This is grounded in the same bounds the generator used to produce
the data, not an arbitrary guess, and doesn't apply to vitals with no alt
unit (heart rate, blood pressure, glucose), which just get the canonical
unit imputed directly.

**Diagnosis-name embedded-date artifact — flagged, not silently
special-cased:** notes render past-history entries as `"{condition} (dx
{onset date})"` (see `generate_notes.py`'s PMH section), and extraction
sometimes captures that whole string — onset date included — as the
diagnosis name. `normalize.py` strips the trailing `(dx ...)` parenthetical
as part of getting to a canonical name, but the date itself is exactly the
kind of visit-adjacent field Stage C's redaction step is supposed to
target, and it's currently only reachable by string-matching inside a name
field rather than as its own structured field. **Upstream fix worth
making:** have `extractor.py`'s tool schema capture diagnosis onset date as
its own optional field, so redaction (and any future analysis) doesn't
depend on it happening to be embedded in free text it can find and strip.
Not fixed now — Stage B is marked complete and the fix is more invasive
than a Stage C concern — but should be revisited before this pipeline is
called done.

## Stage C — redact

**The problem:** normalized records carry three PHI-standin fields —
`patient_name`, `date_of_birth`, `note_date` (visit date) — that
`extractor.py` was deliberately built to capture as concrete structured
fields for this step to act on (see that module's docstring). No MRN or
address field exists anywhere in this pipeline's extraction output, so
there's nothing to redact there.

**`patient_name` — dropped entirely, not masked.** Considered replacing the
value with something like `"[REDACTED]"` to preserve a fixed record shape,
but there's no downstream consumer that benefits from a field being
present-but-masked — Stage D's JSONL pairs are built from clinical content
(diagnoses/medications/vitals), and an inert placeholder token would just be
something the model has to learn to ignore. If evidence that redaction ran
is needed, the run's printed summary serves that better than tokens left in
the data.

**`date_of_birth` and `note_date` — date-shifted, not stripped.** First
pass dropped these outright, same as `patient_name`. Corrected: unlike a
name, dates carry clinical/temporal meaning worth keeping for a model that
might reasonably need "patient age at time of visit" or intervals between
visits — deleting them destroys that signal. But leaving true dates intact
makes re-identification meaningfully easier (dates are one of HIPAA Safe
Harbor's 18 identifying fields), so each date is shifted by a random offset
instead of kept verbatim or removed.

**Offsets are seeded per patient, per *category* — not per field.** First
pass seeded `sha256(patient_id + field name)`, giving `date_of_birth` and
`note_date` independent offsets. That looked correct on this dataset (one
field per category — `dob` has only `date_of_birth`, `visit` has only
`note_date` — so per-field and per-category seeding are indistinguishable
today) but was the wrong mechanism: a second visit-type date field added
later (e.g. a medication-authored date, if extraction ever captures one)
would have gotten its *own* independent offset instead of sharing
`note_date`'s, silently breaking the guarantee that all of a patient's
visit dates move together. Corrected to seed on
`sha256(patient_id + category name)`, where `SHIFTED_FIELDS` maps each
field to a category (`"date_of_birth" -> "dob"`, `"note_date" -> "visit"`);
a future field just needs mapping into the existing `"visit"` category to
inherit its shift, rather than needing new logic.

`dob_shift` and `visit_shift` are independent per patient — a single
shared per-patient offset would let anyone who recovers one true date
(e.g. a known birthdate) trivially unshift every other date for that
patient by subtraction; independent category shifts mean recovering one
category's date reveals nothing about the other's. Verified on the
10-record dev sample: every record's `dob_shift` differs from its
`visit_shift`, and reruns produce identical output (offsets are derived
from a hash, not persisted state).

Shift range is +/- 365 days: large enough that the shifted date isn't
trivially close to the real one, small enough to keep the date in roughly
the right season/year for anyone eyeballing the data. This is a
portfolio-scale choice, not a HIPAA Safe Harbor compliance claim — Safe
Harbor requires year-only generalization for patients 90+, which doesn't
apply to this dataset's ages, so day-level shifting is sufficient here.

**`patient_id` (a synthetic UUID) is kept, not redacted or reassigned.** It
doesn't derive from or correlate with any real identifier, so it carries no
re-identification risk by itself, and it's the join key that lets a record
be traced back through generation -> extraction -> curation during
development. Reassigning it would break that traceability for no privacy
benefit.

**Redaction scope: `redact.py` only touches the notes -> extraction ->
curation branch, not `patient_features.csv`.** `patient_features.csv` is
Stage A's flattened FHIR feature table — a separate, parallel artifact
demonstrating the FHIR-source-of-truth/flatten skill, and it's never
consumed downstream by Stage D/E fine-tuning (only the redacted
notes/extraction branch is). It deliberately isn't run through `redact.py`:
extending redaction to a wide tabular CSV (raw `given_name`/`family_name`/
`birth_date` columns, plus per-vital FHIR dates baked directly into flat
columns rather than a nested structure) is a meaningfully different
problem than redacting the JSONL record branch, and nothing currently
reads that CSV downstream, so there's no fine-tuning-data leakage risk from
leaving it raw.

That said, it *is* a user-facing artifact — the app's Stage A page will
display it once Step 9 wires up the app's data display (see Working plan).
**Action item for Step 9:** when `patient_features.csv` (or any view of its
contents) is rendered in the app, add a one-line caption stating it's raw,
unredacted Stage A demonstration data, distinct from the redacted branch
that actually feeds training — so a viewer doesn't mistake the visible raw
names/birthdates for what the model was actually trained on. Not added now
because the app doesn't render this table yet (Stage A's page is still the
unbuilt stub); tracked here so it isn't forgotten by the time that page is
built.
