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

## Stage A/encode — scikit-learn-ready feature matrix

**Added after Step 9 was already complete**, in response to a fair review
question: the flattened feature table, on its own, only demonstrates
"denormalize nested FHIR into a flat CSV." That's a real step, but it's not
the same skill as "prepare FHIR data for a classical ML model" — which is
what tying this branch back to a scikit-learn workflow actually claims.
`generation/encode_features.py` closes that gap so the claim is earned, not
implied, turning "flattened for display" into "flattened for machine
learning."

**The transforms, and why each (not just that each was used):**

- **`gender` → `OneHotEncoder` + explicit `gender_unknown` flag.** A naive
  `male=0, female=1` integer map would let the model compute
  `female - male = 1` and potentially learn a pattern along that meaningless
  number line. One-hot gives each category its own independent 0/1 column,
  encoding "these are unordered categories" rather than "points on a
  scale." Missing gender (dropped ~15% of the time by the messiness toggle)
  still zeroes both one-hot columns via `handle_unknown="ignore"` — but a
  review caught that this alone is ambiguous: a lone `0` in `gender_female`
  can't be told apart from "confirmed male" versus "unknown," since both
  produce the same value. Corrected by adding an explicit `gender_unknown`
  indicator column, so "we don't know" is its own visible signal rather
  than something inferred from two other columns both being zero — the
  same pattern `flatten.py` already uses for `{vital}_date_unknown`, now
  applied consistently to the encoding step too.
- **`conditions` / `medications` → `MultiLabelBinarizer`.** These are the
  interesting case: a patient legitimately has *several* diagnoses at once,
  so this is multi-label, not single-category — one-hot cannot represent it.
  One 0/1 column per canonical ICD-10 condition / RxNorm medication
  (imported from `generate_fhir.py`, the single source of truth, with a
  fixed `classes=` so the feature schema is stable regardless of which
  categories happen to appear in a given run). A patient can be 1 in several
  columns simultaneously.
- **vital `*_value` → unit-normalize, then `SimpleImputer`.** Unit
  normalization first (lb→kg, in→cm, °F→°C, using the unit column `flatten`
  emits): a body-weight column mixing `92` (kg) and `138` (lb) is genuinely
  garbage as a feature, so harmonizing units is required feature prep, not
  optional polish — and skipping it would make the "ML-ready" claim false.
  Then mean imputation fills genuinely-missing readings so no cell is blank.

**Trade-offs stated honestly:** mean imputation is the simplest defensible
default — it distorts the distribution less than zero-fill but ignores any
structure in the missingness (a vital may be absent *because* it wasn't
clinically relevant, not at random); a production pipeline might add a
missing-indicator column or a model-based imputer. Identifiers
(`patient_id`/`mrn`/name/`birth_date`) are dropped as non-features
(`patient_id` is re-attached only as a join key), and the `*_unit` columns
are dropped because, after normalization, they're constant (zero-variance)
and carry no signal.

**Deliberately not done: fitting an actual model.** This synthetic data has
no genuine labeled outcome (no readmission flag, no real target), so
training a "predictive model" on it would manufacture a meaningless result —
exactly the kind of overclaiming Stage E's honest framing avoids. The
encoding *is* the demonstrated skill here; a fabricated target would add
nothing real. In the app, both the raw flattened table and this encoded view
are collapsed-by-default expanders on the Stage A page, so this parallel
classical-ML branch is available to a reviewer who wants it without crowding
the primary extraction/curation narrative.

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

**Fixed: 3 specific paraphrase variants were invisible to `rebalance.py`'s
category counting.** At the 100-patient scale, real Haiku extraction output
included three diagnosis phrasings that matched neither a canonical name
nor a known `CONDITION_ABBREVIATIONS` abbreviation: `"Hypertension (HTN)"`,
`"Major depressive disorder"` (missing the canonical's `", single episode,
unspecified"` suffix), and `"Type 2 Diabetes Mellitus"` (missing `"without
complications"`). Left unmatched, records carrying only one of these
phrasings didn't count toward their true diagnosis category in
`rebalance.py`'s `category_counts()`, silently understating that category's
representation. Fixed with a small `_DIAGNOSIS_PARAPHRASES` lookup
(mirroring the existing `_ABBREV_TO_CANONICAL` pattern, checked alongside
it in `normalize_diagnosis_name`) mapping exactly these 3 observed
phrasings to their canonical names — scoped narrowly to what was actually
observed and confirmed unmatched, not a general paraphrase-matching pass.
The rest of the ~32-48-item unmatched-values list seen at this scale
(dosage text without the `"Take ... by mouth"` framing, raw UCUM unit codes
like `lb_av`/`degF`/`in_i`) is unrelated dosage/unit variance, still out of
scope for this pass.

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

That said, it *is* a user-facing artifact — the app's Stage A page displays
it. **Resolved in Step 9:** the app's Stage A page now renders this table
with a caption stating it's raw, unredacted Stage A demonstration data,
distinct from the redacted branch that actually feeds training — so a
viewer doesn't mistake the visible raw names/birthdates for what the model
was actually trained on.

**Fixed: a shifted `note_date` could land after the generation ceiling.**
`generate_fhir.py` generates original onset/effective/authored dates up to
a hardcoded ceiling (`date(2026, 7, 8)`); `redact.py` then shifts by up to
+/-`SHIFT_RANGE_DAYS` (365) with no check against that ceiling afterward.
At the previous 100-patient scale, 20/100 records ended up with a
`note_date` shifted past that date — e.g. a note reading "Visit Date:
2026-12-31" while narrating the visit as already having happened. Not a
correctness or privacy issue (doesn't affect diagnosis/medication/vitals
content, doesn't weaken redaction — the shift still moved the true date),
but a real data-realism gap, previously left unfixed because Stage E had
already completed a real run against the then-current data.

**Fix applied:** `generate_fhir.py` now imports `SHIFT_RANGE_DAYS` directly
from `curation.redact` (a single source of truth, not a duplicated magic
number — verified this creates no circular import, since `redact.py` has
no dependency on `generate_fhir.py`) and generates onset/effective/asserted
dates against `GENERATION_CEILING = TRUE_CEILING - timedelta(days=
SHIFT_RANGE_DAYS)` instead of the raw `TRUE_CEILING`. This is still a fixed
historical ceiling, just correctly reduced by the maximum possible shift —
not a dynamic `date.today()` — so no post-shift date can ever exceed
`TRUE_CEILING` (`2026-07-08`) regardless of shift direction. Verified after
the full pipeline rebuild this fix required: max generated onset/effective
date across all 677 dated resources was `2025-07-05`, safely under the
`2025-07-08` generation ceiling; max shifted `note_date` across all 100
redacted records was `2026-05-10`, safely under the true `2026-07-08`
ceiling.

## Stage C — rebalance

**The problem:** Stage B's extraction, run at realistic scale, produces
diagnosis-category counts shaped by `random.choice` over `generate_fhir.py`'s
condition list, not by any deliberate class balance. A fine-tune trained on
that raw distribution would see some diagnoses several times more often
than others for no reason related to real-world prevalence.

**Decision: oversample (duplicate existing records) rather than downsample.**
With a dataset this size, throwing away over-represented records to match
the rarest category would shrink an already-small training set further —
actively worse for a data-starved fine-tune. Duplicating records that carry
an under-represented diagnosis preserves every original example while
boosting rare categories, at the cost of the duplicated content being an
exact copy (a real overfitting risk per duplicate, not engineered away —
see `rebalance.py`'s docstring). Duplicates are tagged
`rebalance_duplicate_of` with a suffixed `patient_id` (`-dup1`, `-dup2`,
...), both for auditability and because Stage D's split **must** keep every
duplicate in the same split as its original (see the split section below).

**Known, accepted limitation — multi-diagnosis records overshoot.** A record
carrying two diagnoses boosts both when duplicated to fix one of them, even
if the other was already at target. Documented rather than solved with a
set-cover-style selection, which isn't worth the complexity at this scale.

**Known, accepted limitation — a zero-represented category can't be
rebalanced at all.** Oversampling can only amplify what's already present;
a diagnosis with zero existing records has nothing to duplicate. That's
exactly the gap `synthesize.py` exists to fill.

**Confirmed fixed by the normalize.py paraphrase fix above:** before that
fix, records extracted with one of the 3 unmatched phrasings didn't count
toward their true category here, occasionally understating a category down
to zero when it wasn't actually zero (observed previously: `"Hypertension
(HTN)"` phrasing left `Essential (primary) hypertension` looking
zero-represented in one run). Re-verified on the rebuilt 100-patient
dataset: every one of the 10 canonical categories shows a non-zero,
plausible `before` count (13-22 records) in `category_counts()`'s output —
no phantom zero/singleton categories from mis-normalized diagnosis names
remain.

## Stage C — synthesize

**The problem:** whichever diagnosis categories `rebalance.py` leaves at
zero can't be fixed by duplication — a category with no existing record has
no record to amplify.

**Decision: LLM generation (Anthropic API, cache-first, same pattern as
`extractor.py`), not a hand-written template record.** A hand-authored
record for the one missing category at dev scale would have been faster and
free, but rejected because it doesn't scale (every zero-count category
found needs its own manual record — exactly the kind of one-off work this
pipeline is meant to automate) and because an LLM can reason about
clinically plausible comorbidities and medications in a way a static
template can't without re-deriving a small clinical knowledge base by hand.

**Division of labor: the model picks *which*, code fills in *how*.** The
model chooses comorbid diagnoses/medications/vitals, enum-constrained to
`generate_fhir.py`'s own canonical tables — but the model never invents unit
strings or dosage text; code fills those in canonically once the model
picks a name. This guarantees synthesized records land in the dataset
already `normalize.py`-canonical, rather than trusting the model to
reproduce exact canonical strings verbatim. `patient_id`/`date_of_birth`/
`note_date` are also generated in code (deterministic, seeded per record),
since there's no clinical judgment in picking a plausible birth date and
keeping them out of the API call keeps the cache key focused on the part
that actually needs LLM reasoning.

**Only fills actual gaps.** Never generates more than `target -
current_count` for a category, and only for categories still at zero after
rebalance — it doesn't round out every category to a nicer number. At the
current 100-patient scale, every category already has organic
representation after rebalancing, so this step runs and correctly makes
zero API calls; the mechanism was exercised (and produced 3 new records) at
the original 10-patient dev scale, when `Hyperlipidemia, unspecified` had
zero records.

Every synthesized record is tagged `"synthesized": true` /
`"synthesized_category"` for traceability, and — same overshoot trade-off
`rebalance.py` documents — the model's comorbidity choices can push an
already-satisfied category further past target as a side effect. Not
engineered away, for the same reason: avoiding it would mean either
clinically unrealistic single-diagnosis records or a more complex
constrained-generation setup neither justified at this scale.

## Stage D — split

**The problem:** Stage C's `rebalance.py` produces duplicate records that
are near-identical copies of an existing record (marked
`rebalance_duplicate_of`) — not independent patients. A random 80/20 split
over raw records could easily put a duplicate in val while its original
sits in train, silently leaking train-seen content into validation.

**Decision: split by original patient identity, not raw record.** Every
record belonging to the same original patient (the original plus every
`-dupN` copy) is treated as one group and assigned to a single split,
always. `synthesize.py`'s output has no matching note in
`clinical_notes.jsonl` (Resolved decisions #8), so those records are
excluded from the split entirely rather than force-fit into a differently-
shaped instruction — a small, documented gap, not a silent one.

**Consequence, stated honestly:** because group sizes vary (a patient with
duplicates forms a group of 2–3; everyone else is a group of 1), splitting
by group only *approximates* an 80/20 *record* ratio, not an exact one.
Anti-leakage correctness takes priority over hitting an exact percentage.
Groups are assigned in first-seen file order, not an additional random
shuffle — patient UUIDs are already randomly generated in Stage A, so the
file's existing order carries no structure worth correcting for.

## Stage D — format into JSONL

**Instruction/response shape.** `instruction` is the patient's redacted note
text; `response` is a JSON-formatted **string** (not a nested object)
containing `date_of_birth`/`diagnoses`/`medications`/`vitals` — a string
because fine-tuning trains the model to generate *text*, and the target it
must learn to produce is the serialized JSON itself, matching how it would
actually need to emit structured output at inference time. Pipeline
bookkeeping (`patient_id`, `rebalance_duplicate_of`, ...) is dropped from
`response` — it's metadata, not something an extraction model should learn
to predict. `note_date` is excluded (it was never an extraction target,
just pipeline-assigned metadata about when the note was generated) but
`date_of_birth` is included, since it *was* one of `extractor.py`'s actual
LLM-extracted fields.

**Note redaction via targeted find-and-replace, not generic de-identification**
— see Resolved decisions #8 for the full rationale (generic free-text PHI
inference is a hard, out-of-scope research problem; this repo already knows
the exact ground-truth strings it inserted, so redaction is substitution,
not inference). All of a patient's known PHI strings and dates are replaced
in a **single compiled regex pass**, not sequential `.replace()` calls —
sequential replacement risks a later rule matching text an earlier rule just
produced (e.g. one date's shifted output happening to equal another field's
original search string), a real correctness bug for simultaneous
substitution, not just a style preference.

**Leakage check is per-patient, not corpus-wide.** At 100-patient scale, a
corpus-wide "does this real string appear anywhere in the output" scan
produced a false positive: patient A's real DOB coincidentally equaled
patient B's own correctly-shifted DOB (a birthday-paradox effect from 100
independent per-patient offsets across a ~66-year range). That collision
discloses nothing about patient A. The check that actually matters is
scoped per patient — does record X's own real PHI appear in record X's own
example — which is what the current implementation does.

## Stage E — train

**Base model: Qwen2.5-0.5B-Instruct**, chosen over TinyLlama-1.1B-Chat
(older, weaker instruction-following, needlessly larger for worse quality)
and SmolLM2-360M-Instruct (smaller, but 0.5B wasn't a meaningful
download/runtime cost on this hardware, and the larger model gives the
fine-tune a better chance of actually picking up the extraction pattern
rather than fighting base capability). Already instruction-tuned, so the
fine-tune only needs to shift behavior toward this specific extraction
format rather than teach instruction-following from scratch.

**No 4-bit/8-bit quantization.** `bitsandbytes` doesn't support MPS well;
skipped rather than fighting a poorly-supported path. At 0.5B parameters,
fp16 LoRA fits comfortably without it.

**LoRA config deliberately modest:** `r=8, alpha=16` (standard `alpha=2r`
heuristic), attention projections only (`q/k/v/o_proj`), not the MLP. With
only 139 training examples, a higher rank or MLP-inclusive target set would
add trainable capacity this dataset can't productively use — it would more
easily memorize the specific 139 examples than learn the general pattern.

**A plain PyTorch training loop, not `transformers.Trainer`.** `Trainer`
would work, but a manual loop keeps every step (batching, chat-template loss
masking, MPS device placement, per-epoch loss) directly inspectable in
~100 lines — appropriate for a script whose job is partly to *demonstrate*
the mechanics, not just produce a checkpoint. Loss is masked to the
assistant continuation only (system+user prefix labels set to `-100`) — the
model should learn to *produce* the extraction, not predict the note text
it's reading.

**A real bug was caught and fixed during this build, not shipped silently.**
The first full run produced pure garbage ("API/API/API...") from *both*
models during before/after sample generation. Root-caused to a missing
`model.eval()` call before generation — the model was left in `.train()`
mode (gradient-checkpointing hooks + LoRA dropout active), confirmed by
reproducing the exact garbage output and warnings in isolation, then
confirming the fix resolved it cleanly. A second, unrelated MPS
out-of-memory issue on the first batch (from Qwen2.5's large 151,936-token
vocab making the logits tensor large relative to this model's hidden size)
was fixed with gradient checkpointing and a smaller batch size.

**Honest framing:** small dataset by real ML standards, and val loss
plateaus/ticks up after its lowest epoch while train loss keeps falling —
the textbook overfitting signature at this scale, reported plainly rather
than smoothed over. What this run *does* demonstrate, correctly and on
real hardware: real data loading, correct loss masking, LoRA attachment, a
real training loop with genuinely declining loss, adapter checkpointing,
and a real behavioral difference with vs. without the adapter. Not a claim
of production-quality extraction accuracy.

**Fixed: reproducibility gap and no checkpoint recovery from overfitting.**
Two gaps in the original script: (1) only the per-epoch data-shuffle order
was seeded (`torch.Generator().manual_seed(42)`); the LoRA adapter's own
initial weights (its `A` matrices are Kaiming-uniform initialized from the
*global* RNG — `B` is zero-init, so it doesn't matter, but `A` does) were
not, so two runs with "the same seed" could still start from different
adapter weights. (2) the script only ever kept the *last* epoch's adapter,
even though val loss reliably plateaus/ticks up well before the last epoch
on this dataset size — so the shipped adapter was never actually the best
one produced during the run.

**Fix applied:** `torch.manual_seed(42)` now runs before `get_peft_model`
creates the adapter, fixing gap (1). Every epoch now saves a full
checkpoint to `training_results/checkpoints/epoch_{N}/`; the epoch with the
lowest val_loss is tracked during training, and after all epochs complete,
that checkpoint's weights are reloaded into the model (via PEFT's
`load_adapter` with the existing `"default"` adapter name, which replaces
in place rather than adding a second adapter) *before* generating the
before/after samples and saving the primary `training_results/adapter/` —
so both the samples and the shipped adapter now reflect the best epoch,
not whichever epoch happened to run last. `config.json` records
`best_epoch`, `best_train_loss`, `best_val_loss` alongside the full
per-epoch `loss_history`, so which epoch was selected and why is explicit,
not implicit in "whatever ran last."

**Real numbers from the full pipeline rebuild + retrain (100-patient scale,
112 train / 20 val examples, 906.0s wall clock on MPS):**
train loss declined every epoch (0.1309 -> 0.0238 -> 0.0127 -> 0.0101 ->
0.0068 -> 0.0055); val loss was lowest at **epoch 3 (0.0304)**, after
declining from 0.0548 (epoch 1) and 0.0375 (epoch 2), then ticking back up
through epochs 4-6 (0.0338, 0.0368, 0.0364) — the same mild-overfitting
shape as the prior run, just now with the shipped adapter actually
reloaded from its best point (epoch 3) rather than its last (epoch 6).
Verified: `training_results/checkpoints/` contains 6 valid epoch
checkpoints (`epoch_1`-`epoch_6`, each a complete adapter), the selected
best epoch (3) matches the lowest val_loss in the printed history, and
`training_results/adapter/adapter_model.safetensors` is byte-identical
(confirmed via checksum) to `checkpoints/epoch_3/adapter_model.safetensors`.

## Committing pipeline outputs (`data/`)

**The problem, found during Step 9 polish:** `.gitignore` excluded
`data/generated/`, `data/extracted/`, `data/curated/`, and `data/splits/` as
"regenerated by the app; not committed." But nothing in this pipeline
regenerates them live in a deployed context — Stages B and C's synthesize
step call the Anthropic API, which a public Streamlit deployment has no key
for, and even without that constraint, re-running ~100 Haiku calls on every
visitor's page load would be slow and needlessly costly. Left as-is, the
app's Stage A–D pages would have nothing to display at all in a fresh
deployment.

**Decision: commit the pipeline outputs, same precedent already set for
`training_results/` (Resolved decisions #3).** This data is fully synthetic
(no real PHI, per this project's Non-goals) and small (~1.5 MB total), so
there's no privacy or repo-bloat cost to committing it. The deployed app now
displays this real, committed, previously-generated run — consistent with
how Stage E already works — rather than silently having five-sixths of the
app be a stub in any environment without an API key.
