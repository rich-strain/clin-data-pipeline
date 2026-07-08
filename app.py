"""Streamlit app shell for the clinical data curation pipeline.

Sidebar stepper walks through the five pipeline stages. Each stage is
currently a stub page; functionality is added incrementally per stage.
"""

import streamlit as st

STAGES = [
    ("A — Generate", "Synthetic FHIR patients, flattened feature table, and clinical notes."),
    ("B — Extract", "LLM-based structured extraction from synthetic notes."),
    ("C — Curate", "Normalize, redact, rebalance, and synthesize the extracted data."),
    ("D — Split & Format", "Train/validation split, emitted as instruction/response JSONL."),
    ("E — Train", "LoRA fine-tune script, config, and results."),
]

st.set_page_config(page_title="Clinical Data Curation Pipeline", layout="wide")

if "stage_index" not in st.session_state:
    st.session_state.stage_index = 0

with st.sidebar:
    st.title("Pipeline Stages")
    for i, (label, _) in enumerate(STAGES):
        if st.button(label, key=f"nav_{i}", use_container_width=True):
            st.session_state.stage_index = i

stage_index = st.session_state.stage_index
stage_label, stage_description = STAGES[stage_index]

st.header(stage_label)
st.caption(stage_description)
st.info("Not yet implemented.")
