"""Minimal Streamlit upload and assessment interface; no inference logic."""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Class Quality Evaluator", page_icon="🎓", layout="wide")
st.title("Class Quality Evaluator")
st.caption("MVP: 2-minute batches · 12 frames per full batch · GPT-4o · six quality signals")
st.info("Only upload recordings you are permitted to process. Frames and transcript text are sent to GPT-4o.")

uploaded = st.file_uploader("Upload class recording", type=["mp4", "mov", "mkv", "webm"])
if uploaded:
    upload_identity = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_identity") != upload_identity:
        st.session_state.pop("assessment", None)
        st.session_state["upload_identity"] = upload_identity
    # Tornado logs a 64 MiB static cache-version warning for large media.
    # Preview is optional: do not send large MP4 files to Streamlit media serving.
    MAX_PREVIEW_BYTES = 64 * 1024 * 1024
    if uploaded.size < MAX_PREVIEW_BYTES:
        st.video(uploaded)
    else:
        st.success(f"Video ready for assessment: {uploaded.name}")
        st.info("Preview skipped for large video; click Assess class to process it.")
else:
    st.session_state.pop("assessment", None)

if st.button("Assess class", type="primary", disabled=uploaded is None):
    assert uploaded is not None
    with st.spinner("Processing video, transcribing, assessing windows, scoring class…"):
        try:
            uploaded.seek(0)
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, uploaded, uploaded.type or "video/mp4")},
                timeout=(15, 7200),
            )
            if not response.ok:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"API error {response.status_code}: {detail}")
                st.stop()
            st.session_state["assessment"] = response.json()
        except requests.RequestException as exc:
            st.error(f"Cannot reach prediction API: {exc}")
            st.stop()

result = st.session_state.get("assessment")
if result:
    st.subheader("Overall result")
    a, b, c = st.columns(3)
    a.metric("Class label", result["label"])
    b.metric("Weighted score", f'{result["score"]}/100')
    c.metric("Two-minute batches", result["batch_count"])
    st.write(result["overall_observation"])

    st.subheader("Quality signal scores")
    for item in result["ratings"]:
        with st.expander(f'{item["signal"].title()} — {item["score"]}/100 (weight {item["weight"]}%)'):
            st.write(item["rationale"])
            st.caption(f'Confidence: {item["confidence"]} | Weighted contribution: {item["contribution"]:.2f}')
            if item["supporting_timestamps"]:
                st.write("Evidence times (seconds):", item["supporting_timestamps"])
            if item.get("criterion_deductions"):
                st.markdown("**Criterion fulfilment and deductions**")
                for criterion in item["criterion_deductions"]:
                    st.write(
                        f"{criterion['criterion_id']}: {criterion['status']} "
                        f"(severity: {criterion['issue_severity']}; "
                        f"+{criterion['points_awarded']:.1f}/-{criterion['points_deducted']:.1f} "
                        f"of {criterion['points_possible']:.1f}) — {criterion['observation']}"
                    )
                    if criterion.get("positive_timestamps"):
                        st.caption(f"Positive evidence (s): {criterion['positive_timestamps']}")
                    if criterion.get("negative_timestamps"):
                        st.caption(f"Negative evidence (s): {criterion['negative_timestamps']}")
            if item["limitations"]:
                st.warning(item["limitations"])

    st.subheader("Suggested improvements")
    for recommendation in result["recommendations"]:
        st.write(f"- {recommendation}")
    st.download_button(
        "Export assessment JSON", data=json.dumps(result, indent=2),
        file_name=f'{result["run_id"]}_assessment.json', mime="application/json",
    )
    st.caption("MVP results are rubric-based estimates, not independently validated scores.")
