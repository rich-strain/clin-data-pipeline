# Clinical Data Curation Pipeline

A complete, self-contained, end-to-end clinical data pipeline: synthetic
patient generation, LLM-based extraction, curation, dataset splitting, and a
real training run. **Curation — normalization, redaction, rebalancing,
synthesis — is the centerpiece** and gets the most attention, since it's the
highest-value part of this kind of work.

Everything here is synthetic. No real patient data is used, scraped, or
referenced anywhere in this repo, under any circumstance.

## Why this exists

Real-world medical records are almost always represented as **FHIR**
(structured demographics/conditions/observations/medications) plus
unstructured free-text notes. Most portfolio "clinical NLP" projects skip
straight to the free text and never touch FHIR at all. This pipeline
generates FHIR **first**, as the system of record, then derives everything
else from it — the flattened feature table, the clinical notes, and
ultimately the fine-tuning data — the same shape a real analytics/ML
pipeline actually takes.

## Pipeline stages

| Stage | What it does | Code |
|---|---|---|
| **A — Generate** | Synthetic `Patient`/`Condition`/`Observation`/`MedicationStatement` FHIR bundles; a flatten step projects them into a tabular feature table; a separate generator writes free-text clinical notes. | `generation/` |
| **B — Extract** | LLM-based structured extraction (Anthropic API, cache-first) pulls diagnosis/medication/dosage/vitals/PHI fields out of the notes. This is the raw "LLM dump" — not yet curated. | `extraction/extractor.py` |
| **C — Curate** | Normalize (canonical units/terminology) → Redact (drop name/MRN/address, date-shift DOB/visit dates) → Rebalance (even out diagnosis-category representation by duplication) → Synthesize (LLM-generated new records for categories duplication can't fix). | `curation/` |
| **D — Split & Format** | ~80/20 train/val split, grouped by original patient identity to prevent duplicate-record leakage across the split; emits instruction/response JSONL. | `split.py`, `format_jsonl.py` |
| **E — Train** | A real LoRA fine-tune of Qwen2.5-0.5B-Instruct, run locally on Apple Silicon (MPS) and committed to the repo. | `train_runner.py` |

Every stage's rationale — and the alternatives considered and rejected — is
written up in [`docs/design_decisions.md`](docs/design_decisions.md).

## Running the app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The deployed/demo app **displays committed pipeline output** — it doesn't
call the Anthropic API or run training live. A public Streamlit deployment
has no API key and no GPU, so Stages A–E all show a real, previously-run,
fully-synthetic dataset (see `docs/design_decisions.md` for why this mirrors
the same call already made for Stage E's training run).

## Re-running the pipeline yourself

Requires an `ANTHROPIC_API_KEY` in `.env` (used by Stages B and C's
synthesize step; cheap on Haiku). Each stage is a standalone script; this is
the full chain, in order:

```bash
python generation/generate_fhir.py --n-patients 100 --messy
python generation/flatten.py
python generation/generate_notes.py --messy
python extraction/extractor.py
python curation/normalize.py
python curation/redact.py
python curation/rebalance.py
python curation/synthesize.py
python split.py
python format_jsonl.py
python train_runner.py   # Stage E — real fine-tune, ~15 min on Apple Silicon MPS
```

Each script takes `--in`/`--out` overrides (see `--help`) and reads/writes
the paths under `data/` shown in the table above.

## Results (Stage E)

A real LoRA fine-tune (`r=8, alpha=16`, attention projections only) of
Qwen2.5-0.5B-Instruct on 139 train / 20 val examples, 6 epochs, ~15 minutes
on an M3 Mac:

- Train loss: 0.110 → 0.004. Val loss: 0.053 → 0.031 (best at epoch 3), then
  plateaus/ticks up slightly — the expected small-dataset overfitting
  signature, reported as-is, not smoothed over.
- The fine-tuned adapter is 4.2 MB (~0.22% of the base model's parameters)
  vs. 953 MB for the base model.
- Before/after samples show a real behavioral difference: the base model
  emits flat, off-schema JSON and in one case hallucinates a vitals reading
  that appears nowhere in the source note; the fine-tuned model emits
  correctly-structured, schema-matching JSON with canonical diagnosis names
  and no hallucination.

This is a correct, honest demonstration of LoRA fine-tuning mechanics on a
small dataset — not a claim of production-grade extraction accuracy. Full
numbers and before/after text are in `training_results/` and in the app's
Stage E page.

## PHI/PII handling

All data is synthetic, generated entirely by this repo's own code — nothing
scraped, nothing de-identified from a real source. Within that synthetic
data, Stage C still treats the structured PHI-standin fields (name, DOB,
MRN, address, visit dates) as if they were real: names/MRN/address are
dropped outright, and dates are shifted (never stripped) with independent
per-patient, per-category offsets — see `docs/design_decisions.md` for the
full rationale, including the free-text note redaction pass and its
documented per-patient (not corpus-wide) leakage check.

One exception, by design: `data/generated/patient_features.csv` (Stage A's
flattened FHIR feature table) is **not** redacted — it's a separate,
parallel artifact demonstrating the FHIR-source-of-truth/flatten skill and
is never consumed by Stage D/E fine-tuning. The app labels it as such
wherever it's shown.

## Non-goals

- Not a novel fine-tuning technique — competence and correctness matter more
  than novelty here.
- Not full FHIR spec coverage — four representative resource types
  (`Patient`, `Condition`, `Observation`, `MedicationStatement`) demonstrate
  familiarity without chasing production FHIR integration.
- Not a claim of clinical-grade extraction accuracy — see Results above.
