"""Streamlit app shell for the clinical data curation pipeline.

Sidebar stepper walks through the five pipeline stages, each showing the
actual committed output of that stage (see CLAUDE.md's Resolved decisions
for why pipeline outputs are committed rather than regenerated live: this
app has no Anthropic API key or GPU available in a public deployment, the
same reasoning already applied to Stage E's training results).
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TRAINING_RESULTS = ROOT / "training_results"

STAGES = [
    ("A — Generate", "Synthetic FHIR patients, flattened feature table, and clinical notes."),
    ("B — Extract", "LLM-based structured extraction from synthetic notes."),
    ("C — Curate", "Normalize, redact, rebalance, and synthesize the extracted data."),
    ("D — Split & Format", "Train/validation split, emitted as instruction/response JSONL."),
    ("E — Train", "LoRA fine-tune script, config, and results."),
]


@st.cache_data
def load_jsonl(relpath: str) -> list[dict]:
    path = ROOT / relpath
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_json(relpath: str):
    path = ROOT / relpath
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_csv(relpath: str):
    path = ROOT / relpath
    return pd.read_csv(path) if path.exists() else None


def missing(*relpaths: str) -> bool:
    paths = [ROOT / p for p in relpaths]
    absent = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if absent:
        st.warning(
            "Missing pipeline output(s): " + ", ".join(absent)
            + ". Run the corresponding stage script to generate them."
        )
    return bool(absent)


def find_by_patient(records: list[dict], patient_id: str) -> dict | None:
    return next((r for r in records if r.get("patient_id") == patient_id), None)


def render_stage_a():
    bundles = load_json("data/generated/fhir_bundles.json")
    notes = load_jsonl("data/generated/clinical_notes.jsonl")
    features = load_csv("data/generated/patient_features.csv")

    if missing(
        "data/generated/fhir_bundles.json",
        "data/generated/clinical_notes.jsonl",
        "data/generated/patient_features.csv",
    ):
        return

    st.metric("Patients generated", len(bundles))

    st.subheader("Synthetic FHIR bundle (system of record)")
    st.caption(
        "Patient/Condition/Observation/MedicationStatement resources, generated "
        "directly as FHIR — the tabular feature table below is a *view* derived "
        "from this, not the original source."
    )
    sample_bundle = bundles[0]
    patient_resource = next(
        e["resource"] for e in sample_bundle["entry"] if e["resource"]["resourceType"] == "Patient"
    )
    with st.expander(f"Bundle for patient {patient_resource['id']} ({len(sample_bundle['entry'])} resources)"):
        for entry in sample_bundle["entry"]:
            resource = entry["resource"]
            st.caption(resource["resourceType"])
            st.json(resource, expanded=False)

    st.subheader("Flattened feature table")
    st.caption(
        "Raw, unredacted Stage A demonstration output — one row per patient, "
        "projected from the FHIR bundles above. This table still carries real "
        "synthetic names/DOB and is a separate, parallel artifact never "
        "consumed by Stage D/E training; the redacted branch that actually "
        "feeds fine-tuning is shown in Stages C and D."
    )
    st.dataframe(features, width='stretch')

    st.subheader("Sample clinical note")
    st.caption("Free-text note generated to feed Stage B's extraction — not derived from the FHIR data above.")
    sample_note = find_by_patient(notes, patient_resource["id"])
    if sample_note:
        st.text(sample_note["note_text"])


def render_stage_b():
    notes = load_jsonl("data/generated/clinical_notes.jsonl")
    extractions = load_jsonl("data/extracted/extractions.jsonl")
    cache = load_json("extraction/cache/extraction_cache.json") or {}

    if missing("data/generated/clinical_notes.jsonl", "data/extracted/extractions.jsonl"):
        return

    col1, col2 = st.columns(2)
    col1.metric("Notes extracted", len(extractions))
    col2.metric("Cached extraction responses", len(cache))
    st.caption(
        "Extraction is cache-first, keyed on a hash of the note text — repeated dev "
        "runs iterate against this cache instead of re-hitting the Anthropic API."
    )

    st.subheader("Note → structured extraction")
    st.caption(
        "The model preserves the note's own wording on purpose (abbreviations, "
        "shorthand, missing units) — Stage C is what cleans this up."
    )
    if notes and extractions:
        sample_extraction = extractions[0]
        sample_note = find_by_patient(notes, sample_extraction["patient_id"])
        left, right = st.columns(2)
        with left:
            st.caption("Source note")
            st.text(sample_note["note_text"] if sample_note else "(no matching note)")
        with right:
            st.caption("Raw LLM extraction")
            st.json(sample_extraction)


def render_stage_c():
    extractions = load_jsonl("data/extracted/extractions.jsonl")
    normalized = load_jsonl("data/curated/normalized.jsonl")
    redacted = load_jsonl("data/curated/redacted.jsonl")
    rebalanced = load_jsonl("data/curated/rebalanced.jsonl")
    synthesized = load_jsonl("data/curated/synthesized.jsonl")

    if missing(
        "data/extracted/extractions.jsonl",
        "data/curated/normalized.jsonl",
        "data/curated/redacted.jsonl",
        "data/curated/rebalanced.jsonl",
        "data/curated/synthesized.jsonl",
    ):
        return

    tab_norm, tab_redact, tab_rebalance, tab_synth = st.tabs(
        ["Normalize", "Redact", "Rebalance", "Synthesize"]
    )

    with tab_norm:
        st.caption(
            "Diagnosis abbreviations, dosage shorthand, and vital units are matched "
            "against generate_fhir.py's own canonical tables — an exact lookup, "
            "not fuzzy NLP, since this is a closed-vocabulary synthetic dataset."
        )
        sample_id = extractions[0]["patient_id"]
        before = find_by_patient(extractions, sample_id)
        after = find_by_patient(normalized, sample_id)
        left, right = st.columns(2)
        left.caption("Before (raw extraction)")
        left.json(before)
        right.caption("After (normalized)")
        right.json(after)

    with tab_redact:
        st.caption(
            "`patient_name`/`mrn`/`address` are dropped outright. `date_of_birth` "
            "and `note_date` are date-shifted (not stripped) so age/interval "
            "signal survives — each shifted independently, seeded per category "
            "(dob vs. visit) so recovering one date can't unshift the other."
        )
        sample_id = normalized[0]["patient_id"]
        before = find_by_patient(normalized, sample_id)
        after = find_by_patient(redacted, sample_id)
        left, right = st.columns(2)
        left.caption("Before (normalized)")
        left.json(before)
        right.caption("After (redacted)")
        right.json(after)

    with tab_rebalance:
        st.caption(
            "Duplicates existing records to even out diagnosis-category "
            "representation — oversampling, not downsampling, so an already-"
            "small dataset isn't shrunk further. Duplicates are tagged "
            "`rebalance_duplicate_of` and Stage D keeps every duplicate in the "
            "same train/val split as its original to prevent leakage."
        )

        def category_counts(records):
            counts: dict[str, int] = {}
            for r in records:
                for dx in r.get("diagnoses", []):
                    counts[dx["name"]] = counts.get(dx["name"], 0) + 1
            return counts

        before_counts = category_counts(redacted)
        after_counts = category_counts(rebalanced)
        all_categories = sorted(set(before_counts) | set(after_counts))
        counts_df = pd.DataFrame(
            {
                "category": all_categories,
                "before": [before_counts.get(c, 0) for c in all_categories],
                "after": [after_counts.get(c, 0) for c in all_categories],
            }
        )
        st.dataframe(counts_df, width='stretch', hide_index=True)
        n_dups = sum(1 for r in rebalanced if "rebalance_duplicate_of" in r)
        st.caption(f"{n_dups} duplicate record(s) added ({len(redacted)} → {len(rebalanced)}).")

    with tab_synth:
        st.caption(
            "Fills any diagnosis category still at zero after rebalancing — "
            "duplication can only amplify an existing record, never manufacture "
            "one from nothing. Uses the Anthropic API (cache-first) to pick "
            "clinically plausible comorbidities, medications, and vitals, "
            "enum-constrained to generate_fhir.py's own tables so synthesized "
            "records land already normalize.py-canonical."
        )
        n_synth = sum(1 for r in synthesized if r.get("synthesized"))
        if n_synth == 0:
            st.info(
                f"At the current {len(rebalanced)}-record scale, every diagnosis "
                "category already had organic representation after rebalancing — "
                "synthesize.py detected zero deficit and made zero API calls. "
                "(At the original 10-patient dev sample, this step generated 3 "
                "new `Hyperlipidemia, unspecified` records to fill exactly this "
                "kind of gap — the mechanism is exercised there, just not needed "
                "at this scale.)"
            )
        else:
            st.metric("Synthesized records", n_synth)
            sample = next(r for r in synthesized if r.get("synthesized"))
            st.json(sample)


def render_stage_d():
    train = load_jsonl("data/splits/train.jsonl")
    val = load_jsonl("data/splits/val.jsonl")

    if missing("data/splits/train.jsonl", "data/splits/val.jsonl"):
        return

    total = len(train) + len(val)
    col1, col2, col3 = st.columns(3)
    col1.metric("Train examples", len(train))
    col2.metric("Val examples", len(val))
    col3.metric("Train / val split", f"{len(train) / total:.0%} / {len(val) / total:.0%}" if total else "—")

    st.caption(
        "Split by *original patient identity*, not raw record — every "
        "rebalance.py duplicate is kept in the same split as its original, so "
        "no near-identical content leaks across the train/val boundary. This "
        "means the record-level ratio only approximates 80/20; grouping "
        "correctness takes priority over hitting an exact percentage."
    )

    st.subheader("Sample training example")
    st.caption(
        "`instruction` is the patient's redacted note text; `response` is the "
        "target JSON string the model is trained to produce."
    )
    if train:
        sample = train[0]
        st.text_area("instruction", sample["instruction"], height=200, disabled=True)
        st.code(sample["response"], language="json")


def render_stage_e():
    config = load_json("training_results/config.json")
    samples = load_json("training_results/samples.json")
    loss_curve = TRAINING_RESULTS / "loss_curve.png"
    adapter_file = TRAINING_RESULTS / "adapter" / "adapter_model.safetensors"

    if missing("training_results/config.json", "training_results/samples.json"):
        return

    st.caption(
        f"LoRA fine-tune of {config['base_model']} on Stage D's train/val JSONL, "
        "run locally on Apple Silicon (MPS) and committed here — Streamlit Cloud "
        "has no GPU to run this live."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Train / val examples", f"{config['train_examples']} / {config['val_examples']}")
    col2.metric("Epochs", config["epochs"])
    col3.metric("Wall clock", f"{config['training_seconds'] / 60:.1f} min")

    if loss_curve.exists():
        st.image(str(loss_curve), caption="Train/val loss by epoch")

    hist = config["loss_history"]
    loss_df = pd.DataFrame(
        {"epoch": hist["epoch"], "train_loss": hist["train_loss"], "val_loss": hist["val_loss"]}
    )
    st.dataframe(loss_df, width='stretch', hide_index=True)
    st.caption(
        "Val loss plateaus/ticks up slightly after epoch 3 while train loss keeps "
        "falling — the expected signature of mild overfitting on a "
        f"{config['train_examples']}-example dataset, reported plainly rather "
        "than smoothed over."
    )

    if adapter_file.exists():
        adapter_mb = adapter_file.stat().st_size / (1024 * 1024)
        st.metric("LoRA adapter size", f"{adapter_mb:.1f} MB")
        st.caption(
            "vs. ~953 MB for the base model (not committed — re-downloadable "
            "from Hugging Face; only the adapter is repo-worthy)."
        )

    st.subheader("Before / after: base model vs. fine-tuned adapter")
    for i, sample in enumerate(samples):
        with st.expander(f"Val example {i + 1}"):
            st.text_area("Instruction", sample["instruction"], height=150, disabled=True, key=f"instr_{i}")
            st.caption("Ground truth")
            st.code(sample["ground_truth"], language="json")
            st.caption("Base model output")
            st.code(sample["base_model_output"])
            st.caption("Fine-tuned output")
            st.code(sample["fine_tuned_output"], language="json")

    st.info(
        "Honest framing: this demonstrates correct LoRA fine-tuning mechanics — "
        "real data, correct loss masking, a real training loop with genuinely "
        "declining loss, and a real behavioral difference with vs. without the "
        "adapter — on a genuinely small dataset. It is not a claim of "
        "production-quality extraction accuracy."
    )


RENDERERS = [render_stage_a, render_stage_b, render_stage_c, render_stage_d, render_stage_e]

st.set_page_config(page_title="Clinical Data Curation Pipeline", layout="wide")

if "stage_index" not in st.session_state:
    st.session_state.stage_index = 0

with st.sidebar:
    st.title("Pipeline Stages")
    for i, (label, _) in enumerate(STAGES):
        if st.button(label, key=f"nav_{i}", width='stretch'):
            st.session_state.stage_index = i

stage_index = st.session_state.stage_index
stage_label, stage_description = STAGES[stage_index]

st.header(stage_label)
st.caption(stage_description)
RENDERERS[stage_index]()
