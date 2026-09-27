"""
Phase 8 — Streamlit UI for the Marketing Intelligence Agent.

Run with:
    streamlit run streamlit_app.py

Design notes:
  - Uses app.agent.orchestrator.run_new() directly — the exact same function
    the CLI (app/main.py) calls. No duplicated pipeline logic.
  - Live per-stage progress works via the `progress_callback` parameter
    added to run_pipeline()/run_new(): Streamlit renders UI updates as soon
    as they happen within the same script execution, even mid-blocking-call,
    so a callback that updates an st.status() container gives real-time
    feedback without needing polling or a background thread.
  - A completed run's ResearchState is kept in st.session_state so
    switching between result tabs doesn't require re-running the (slow,
    real-API-calling) pipeline.
  - Previously saved runs (data/state_*.json) can be loaded and viewed
    without spending any API credit or time re-running — useful given how
    long a live run can take on a rate-limited free tier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app.agent.orchestrator import run_new
from app.agent.state import ResearchState
from app.config import load_settings

st.set_page_config(page_title="Marketing Intelligence Agent", page_icon="📊", layout="wide")

_STAGE_LABELS = {
    "init": "Initializing run",
    "research": "Researching public web sources",
    "processing": "Building knowledge base",
    "extraction": "Extracting evidence-backed claims",
    "analysis": "Analyzing business, competitors, customers, content & market",
    "strategy": "Generating marketing strategy",
    "campaigns": "Generating campaign concepts",
    "action_plan": "Building the action plan",
    "validation": "Validating evidence & strategy integrity",
}


def _friendly_stage(stage: str) -> str:
    return _STAGE_LABELS.get(stage, stage.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Sidebar: load a previous run without spending API credit / time
# ---------------------------------------------------------------------------

def _list_saved_runs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("state_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


with st.sidebar:
    st.header("Previous runs")
    try:
        settings_preview = load_settings()
        saved_runs = _list_saved_runs(settings_preview.data_dir)
    except RuntimeError:
        saved_runs = []
        settings_preview = None

    if saved_runs:
        labels = [p.stem.replace("state_", "") for p in saved_runs]
        selected_idx = st.selectbox(
            "Load a saved run to view (free, no API calls)",
            options=range(len(saved_runs)),
            format_func=lambda i: labels[i],
        )
        if st.button("Load selected run"):
            st.session_state["result_state"] = ResearchState.load(saved_runs[selected_idx])
            st.session_state["run_error"] = None
    else:
        st.caption("No saved runs found yet.")

    if settings_preview is not None:
        st.divider()
        st.caption(f"LLM provider: **{settings_preview.llm_provider}**")


# ---------------------------------------------------------------------------
# Main: input form
# ---------------------------------------------------------------------------

st.title("📊 Marketing Intelligence Agent")
st.caption(
    "Evidence-backed marketing research and strategy generation — no external "
    "research APIs, every claim traceable to a real public source."
)

with st.form("run_form"):
    col1, col2 = st.columns(2)
    with col1:
        business_name = st.text_input("Business name", placeholder="e.g. Nanavati Toyota")
        industry = st.text_input("Industry", placeholder="e.g. Automobile")
    with col2:
        location = st.text_input("Location", placeholder="e.g. Surat, Gujarat")
        objective = st.text_input("Marketing objective", placeholder="e.g. Increase customer acquisition")

    homepage_url = st.text_input(
        "Official website (optional, but strongly recommended)",
        placeholder="https://example.com/",
        help="Without this, the business itself won't be directly researched — only "
        "competitors/content/customer/market sources Claude suggests as candidates.",
    )

    submitted = st.form_submit_button("Run Marketing Analysis", type="primary")


# ---------------------------------------------------------------------------
# Run the pipeline, with live per-stage progress
# ---------------------------------------------------------------------------

if submitted:
    if not (business_name and industry and location and objective):
        st.error("Business name, industry, location, and objective are all required.")
    else:
        try:
            settings = load_settings()
        except RuntimeError as exc:
            st.error(f"Configuration error: {exc}")
            st.stop()

        status_box = st.status("Starting pipeline...", expanded=True)
        log_placeholder = status_box.empty()
        seen_stage_count = 0

        def _on_progress(state: ResearchState) -> None:
            global seen_stage_count
            new_entries = state.stage_log[seen_stage_count:]
            seen_stage_count = len(state.stage_log)
            for entry in new_entries:
                icon = {"running": "⏳", "success": "✅", "failed": "❌", "skipped": "⏭️"}.get(
                    entry.status.value, "•"
                )
                status_box.write(f"{icon} **{_friendly_stage(entry.stage)}** — {entry.detail}")
            if new_entries:
                status_box.update(label=f"Running: {_friendly_stage(new_entries[-1].stage)}...")

        try:
            result_state = run_new(
                business_name=business_name,
                industry=industry,
                location=location,
                objective=objective,
                homepage_url=homepage_url,
                settings=settings,
                progress_callback=_on_progress,
            )
            status_box.update(label="Pipeline complete!", state="complete", expanded=False)
            st.session_state["result_state"] = result_state
            st.session_state["run_error"] = None
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI rather than crash it
            status_box.update(label="Pipeline failed", state="error")
            st.session_state["run_error"] = str(exc)
            st.error(f"The run failed: {exc}")


# ---------------------------------------------------------------------------
# Results rendering — shared by both a fresh run and a loaded saved run
# ---------------------------------------------------------------------------

def _render_results(state: ResearchState) -> None:
    st.divider()
    st.header(f"Results: {state.input.business_name}")

    ok_docs = sum(1 for d in state.raw_documents if d.get("ok"))
    fact_claims = sum(1 for c in state.claims if c.get("claim_type") == "fact")
    obs_claims = sum(1 for c in state.claims if c.get("claim_type") == "observation")
    campaigns = state.strategy.get("campaigns", {}).get("campaigns", [])
    action_plan = state.strategy.get("action_plan", {})
    total_actions = sum(len(p.get("actions", [])) for p in action_plan.get("plan", []))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Documents fetched", f"{ok_docs}/{len(state.raw_documents)}")
    m2.metric("Facts extracted", fact_claims)
    m3.metric("Observations", obs_claims)
    m4.metric("Campaigns", len(campaigns))
    m5.metric("Action items", total_actions)

    tabs = st.tabs(
        ["Sources", "Claims", "Analysis", "Strategy", "Campaigns", "Action Plan", "Validation"]
    )

    with tabs[0]:
        for doc in state.raw_documents:
            icon = "✅" if doc.get("ok") else "❌"
            with st.expander(f"{icon} [{doc.get('category')}] {doc.get('url')}"):
                if doc.get("ok"):
                    st.write(f"**Title:** {doc.get('title')}")
                    st.write((doc.get("main_text") or "")[:500] + "...")
                else:
                    st.write(f"**Unavailable:** {doc.get('error')}")

    with tabs[1]:
        for claim in state.claims:
            st.markdown(
                f"**[{claim.get('claim_type', '').upper()}]** {claim.get('claim')}  \n"
                f"> *\"{claim.get('evidence')}\"* — [{claim.get('source_url')}]({claim.get('source_url')})"
            )

    with tabs[2]:
        for dimension, result in state.analysis.items():
            st.subheader(dimension.title())
            st.write(result.get("summary", ""))
            for rec in result.get("recommendations", []):
                st.markdown(f"- **Recommendation:** {rec.get('recommendation')} — _{rec.get('reason')}_")

    with tabs[3]:
        positioning = state.strategy.get("positioning", {})
        st.subheader("Positioning")
        st.write(positioning.get("statement", "(none)"))
        for section_name in ("target_audience", "messaging", "channels", "strategic_priorities"):
            items = state.strategy.get(section_name, [])
            if items:
                st.subheader(section_name.replace("_", " ").title())
                for item in items:
                    text = next((v for k, v in item.items() if k != "claim_ids"), "")
                    st.markdown(f"- {text}")

    with tabs[4]:
        if not campaigns:
            st.info("No campaigns survived evidence validation for this run.")
        for c in campaigns:
            with st.expander(c.get("name", "Unnamed campaign")):
                st.write(f"**Objective:** {c.get('objective')}")
                st.write(f"**Audience:** {c.get('audience')}")
                st.write(f"**Message:** {c.get('message')}")
                st.write(f"**Reasoning:** {c.get('reasoning')}")

    with tabs[5]:
        for phase in action_plan.get("plan", []):
            st.subheader(phase.get("phase", ""))
            for action in phase.get("actions", []):
                st.markdown(f"- **{action.get('action')}** ({action.get('channel')}) — {action.get('reason')}")

    with tabs[6]:
        validation = state.validation_report
        status_color = "green" if validation.get("status") == "success" else "red"
        st.markdown(f"**Status:** :{status_color}[{validation.get('status', 'unknown')}]")
        if validation.get("errors"):
            st.error("\n".join(validation["errors"]))
        if validation.get("warnings"):
            st.warning("\n".join(validation["warnings"]))

    st.divider()
    st.download_button(
        "⬇ Download full run data (JSON)",
        data=json.dumps(json.loads(state.model_dump_json()), indent=2),
        file_name=f"state_{state.run_id}.json",
        mime="application/json",
    )


if st.session_state.get("result_state") is not None:
    _render_results(st.session_state["result_state"])
