"""
Main orchestration layer for the Marketing Intelligence Agent.

This module connects the existing research, processing, analysis, strategy,
and validation components into one executable pipeline.

Pipeline:

    input
      ↓
    autonomous discovery
      ↓
    web research
      ↓
    document processing
      ↓
    claim extraction
      ↓
    analysis
      ↓
    strategy
      ↓
    campaigns
      ↓
    action plan
      ↓
    validation
      ↓
    persisted ResearchState

The orchestrator intentionally contains very little business logic.
Individual modules remain responsible for their own domain logic.

This makes the system:
    - observable
    - resumable
    - testable
    - easier to debug
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from app.agent.state import (
    ResearchState,
    StageStatus,
)
from app.analysis.business_analysis import analyze_business
from app.analysis.competitor_analysis import analyze_competitors
from app.analysis.content_analysis import analyze_content
from app.analysis.customer_analysis import analyze_customers
from app.analysis.market_analysis import analyze_market
from app.config import Settings
from app.processing.extractor import extract_claims_from_documents
from app.processing.markdown_processor import (
    build_knowledge_post,
    load_knowledge_documents,
    save_knowledge_document,
    save_raw_documents,
)
from app.research.auto_research import run_autonomous_research
from app.strategy.action_plan import generate_action_plan
from app.strategy.campaign_generator import generate_campaigns
from app.strategy.strategist import generate_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_state(
    state: ResearchState,
    settings: Settings,
) -> Path:
    """Persist the current state after every major stage."""

    path = settings.data_dir / f"state_{state.run_id}.json"
    state.save(path)

    return path


def _mark_success(
    state: ResearchState,
    stage: str,
    detail: str = "",
) -> None:
    """Record successful completion of a stage."""

    state.log_stage(
        stage,
        StageStatus.SUCCESS,
        detail=detail,
    )


def _mark_failure(
    state: ResearchState,
    stage: str,
    exc: Exception,
) -> None:
    """Record a failed stage without hiding the underlying exception."""

    state.log_stage(
        stage,
        StageStatus.FAILED,
        detail=f"{type(exc).__name__}: {exc}",
    )


def _documents_to_posts(
    documents: list[dict[str, Any]],
    business_name: str,
    settings: Settings,
) -> list[Any]:
    """
    Convert raw research documents into persisted knowledge documents.

    Existing markdown_processor functionality remains the single source of
    truth for the storage format.
    """

    posts: list[Any] = []

    for document in documents:
        try:
            post = build_knowledge_post(
                document,
                business_name,
            )

            save_knowledge_document(
                post,
                settings,
            )

            posts.append(post)

        except Exception:
            logger.exception(
                "Failed to convert research document into knowledge document: %s",
                document.get("url"),
            )

    return posts


def _load_knowledge_posts_for_business(
    business_name: str,
    settings: Settings,
) -> list[Any]:
    """
    Load only this business's knowledge documents.

    load_knowledge_documents() reads the entire shared data/knowledge/
    folder, which accumulates documents from EVERY run and EVERY business
    ever processed (e.g. earlier manual testing, or a previous business run
    that was never cleaned up). Without this filter, a second run — even
    for a completely different business — would silently pick up stale,
    unrelated documents and feed them into extraction, analysis, and
    strategy. Filtering by the `business` field every document's
    frontmatter already carries is the minimal fix; a future phase could
    instead scope storage per-run if this needs to be more robust (e.g.
    concurrent runs for the same business).
    """
    all_posts = load_knowledge_documents(settings)
    return [post for post in all_posts if post.get("business") == business_name]


def _safe_analysis(
    name: str,
    fn: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run one analysis module.

    Analysis modules are independent, so one failed dimension should not
    silently corrupt the other dimensions.
    """

    try:
        result = fn(*args, **kwargs)

        if not isinstance(result, dict):
            logger.error(
                "%s returned %s instead of dict.",
                name,
                type(result).__name__,
            )
            return {}

        return result

    except Exception:
        logger.exception(
            "Analysis stage '%s' failed.",
            name,
        )
        return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    state: ResearchState,
    settings: Settings,
    search_fn=None,
    resume: bool = False,
    progress_callback: Optional[Callable[[ResearchState], None]] = None,
) -> ResearchState:
    """
    Execute the complete marketing intelligence pipeline.

    Parameters
    ----------
    state:
        Initial or previously persisted ResearchState.

    settings:
        Shared application configuration.

    search_fn:
        Optional injected search provider used by autonomous discovery.

    resume:
        Reserved for future fine-grained stage resumption. The current
        implementation safely reuses persisted state fields where possible.

    progress_callback:
        Optional callable invoked with the current `state` every time a
        checkpoint is saved (i.e. after every stage starts and finishes).
        Added for Phase 8 (Streamlit UI): a UI can pass a callback that
        updates on-screen stage progress live, during this same blocking
        call — no polling or background thread needed, since Streamlit
        flushes UI updates as soon as they happen within the same script
        run. Optional and backward-compatible: every existing caller
        (the CLI, tests) that doesn't pass this keeps working unchanged.

    Returns
    -------
    ResearchState
        The fully populated state object.
    """

    business = state.input

    def _checkpoint(current_state: ResearchState) -> None:
        _save_state(current_state, settings)
        if progress_callback is not None:
            progress_callback(current_state)

    # ---------------------------------------------------------------
    # Stage 1 — autonomous research
    # ---------------------------------------------------------------

    if not state.raw_documents:
        state.log_stage(
            "research",
            StageStatus.RUNNING,
            detail="Discovering and researching public web sources.",
        )

        _checkpoint(state)

        try:
            research_result = run_autonomous_research(
                business_name=business.business_name,
                industry=business.industry,
                location=business.location,
                objective=business.objective,
                homepage_url=business.homepage_url,
                settings=settings,
                search_fn=search_fn,
            )

            documents = research_result["all_documents"]

            state.raw_documents = documents

            # Persist the raw research layer.
            save_raw_documents(
                documents,
                business.business_name,
                settings,
            )

            _mark_success(
                state,
                "research",
                detail=(
                    f"Discovered {sum(len(v) for v in research_result['discovery'].values())} "
                    f"sources and fetched {len(documents)} documents."
                ),
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "research", exc)
            _checkpoint(state)
            raise

    # ---------------------------------------------------------------
    # Stage 2 — knowledge documents
    # ---------------------------------------------------------------

    # IMPORTANT: filtered to THIS business only — see
    # _load_knowledge_posts_for_business() docstring. Using the unfiltered
    # global folder here previously meant a second run (any business, any
    # time) would see leftover files from earlier runs, wrongly conclude
    # "already processed", and skip converting the current run's actual
    # freshly-fetched documents — silently feeding stale/unrelated data
    # into extraction instead.
    knowledge_posts = _load_knowledge_posts_for_business(business.business_name, settings)

    if not knowledge_posts and state.raw_documents:
        state.log_stage(
            "processing",
            StageStatus.RUNNING,
            detail="Converting raw research into knowledge documents.",
        )

        _checkpoint(state)

        try:
            knowledge_posts = _documents_to_posts(
                state.raw_documents,
                business.business_name,
                settings,
            )

            state.knowledge_documents = [
                {
                    "url": post.get("url"),
                    "category": post.get("category"),
                    "status": post.get("status"),
                    "confidence": post.get("confidence"),
                }
                for post in knowledge_posts
            ]

            _mark_success(
                state,
                "processing",
                detail=f"Processed {len(knowledge_posts)} knowledge documents.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "processing", exc)
            _checkpoint(state)
            raise

    # ---------------------------------------------------------------
    # Stage 3 — claim extraction
    # ---------------------------------------------------------------

    if not state.claims:
        state.log_stage(
            "extraction",
            StageStatus.RUNNING,
            detail="Extracting evidence-backed claims.",
        )

        _checkpoint(state)

        try:
            state.claims = extract_claims_from_documents(
                knowledge_posts,
                settings,
            )

            _mark_success(
                state,
                "extraction",
                detail=f"Extracted {len(state.claims)} evidence-backed claims.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "extraction", exc)
            _checkpoint(state)
            raise

    # ---------------------------------------------------------------
    # Stage 4 — analysis
    # ---------------------------------------------------------------

    if not state.analysis:
        state.log_stage(
            "analysis",
            StageStatus.RUNNING,
            detail="Analyzing business, competitors, customers, content, and market.",
        )

        _checkpoint(state)

        claims = state.claims

        try:
            state.analysis = {
                "business": _safe_analysis(
                    "business",
                    analyze_business,
                    claims,
                    settings,
                    business.objective,
                ),
                "competitors": _safe_analysis(
                    "competitors",
                    analyze_competitors,
                    claims,
                    settings,
                    business.objective,
                ),
                "customers": _safe_analysis(
                    "customers",
                    analyze_customers,
                    claims,
                    settings,
                    business.objective,
                ),
                "content": _safe_analysis(
                    "content",
                    analyze_content,
                    claims,
                    settings,
                    business.objective,
                ),
                "market": _safe_analysis(
                    "market",
                    analyze_market,
                    claims,
                    settings,
                    business.objective,
                ),
            }

            _mark_success(
                state,
                "analysis",
                detail="Completed five marketing analysis dimensions.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "analysis", exc)
            _checkpoint(state)
            raise

    # ---------------------------------------------------------------
    # Stage 5 — strategy
    # ---------------------------------------------------------------

    if not state.strategy:
        state.log_stage(
            "strategy",
            StageStatus.RUNNING,
            detail="Generating evidence-backed marketing strategy.",
        )

        _checkpoint(state)

        try:
            state.strategy = generate_strategy(
                business_name=business.business_name,
                industry=business.industry,
                location=business.location,
                objective=business.objective,
                claims=state.claims,
                analysis=state.analysis,
                settings=settings,
            )

            _mark_success(
                state,
                "strategy",
                detail="Marketing strategy generated.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "strategy", exc)
            _checkpoint(state)
            raise

    # ---------------------------------------------------------------
    # Stage 6 — campaigns
    # ---------------------------------------------------------------

    campaigns: dict[str, Any] = {}

    if not state.strategy.get("campaigns"):
        state.log_stage(
            "campaigns",
            StageStatus.RUNNING,
            detail="Generating evidence-backed campaign concepts.",
        )

        _checkpoint(state)

        try:
            campaigns = generate_campaigns(
                business_name=business.business_name,
                objective=business.objective,
                claims=state.claims,
                analysis=state.analysis,
                strategy=state.strategy,
                settings=settings,
            )

            state.strategy["campaigns"] = campaigns

            _mark_success(
                state,
                "campaigns",
                detail="Campaign concepts generated.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "campaigns", exc)
            _checkpoint(state)
            raise
    else:
        campaigns = state.strategy.get("campaigns", {})

    # ---------------------------------------------------------------
    # Stage 7 — action plan
    # ---------------------------------------------------------------

    action_plan = state.strategy.get("action_plan")

    if not action_plan:
        state.log_stage(
            "action_plan",
            StageStatus.RUNNING,
            detail="Turning strategy into executable marketing actions.",
        )

        _checkpoint(state)

        try:
            action_plan = generate_action_plan(
                objective=business.objective,
                claims=state.claims,
                strategy=state.strategy,
                campaigns=campaigns,
                settings=settings,
            )

            state.strategy["action_plan"] = action_plan

            _mark_success(
                state,
                "action_plan",
                detail="Action plan generated.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "action_plan", exc)
            _checkpoint(state)
            raise

    # ---------------------------------------------------------------
    # Stage 8 — validation
    # ---------------------------------------------------------------

    if not state.validation_report:
        state.log_stage(
            "validation",
            StageStatus.RUNNING,
            detail="Validating evidence references and strategy integrity.",
        )

        _checkpoint(state)

        try:
            # Validation module is intentionally imported lazily.
            # This keeps the orchestrator compatible with the project while
            # validation.py is being finalized.
            from app.validation.validator import validate_pipeline

            state.validation_report = validate_pipeline(
                claims=state.claims,
                analysis=state.analysis,
                strategy=state.strategy,
                campaigns=campaigns,
                action_plan=action_plan,
            )

            _mark_success(
                state,
                "validation",
                detail="Pipeline validation completed.",
            )

            _checkpoint(state)

        except ImportError:
            # Validation implementation may not exist yet.
            state.validation_report = {
                "status": "pending",
                "warnings": [
                    "Validation module is not yet available."
                ],
            }

            state.log_stage(
                "validation",
                StageStatus.SKIPPED,
                detail="Validation module unavailable.",
            )

            _checkpoint(state)

        except Exception as exc:
            _mark_failure(state, "validation", exc)
            _checkpoint(state)
            raise

    logger.info(
        "Pipeline completed for '%s'.",
        business.business_name,
    )

    return state


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_new(
    *,
    business_name: str,
    industry: str,
    location: str,
    objective: str,
    settings: Settings,
    homepage_url: str = "",
    search_fn=None,
    progress_callback: Optional[Callable[[ResearchState], None]] = None,
) -> ResearchState:
    """
    Create a new ResearchState and run the full pipeline.
    """

    from app.agent.state import BusinessInput

    business_input = BusinessInput(
        business_name=business_name,
        industry=industry,
        location=location,
        objective=objective,
        homepage_url=homepage_url,
    )

    state = ResearchState.new(business_input)

    state.log_stage(
        "init",
        StageStatus.SUCCESS,
        detail="Run created.",
    )

    _save_state(state, settings)
    if progress_callback is not None:
        progress_callback(state)

    return run_pipeline(
        state=state,
        settings=settings,
        search_fn=search_fn,
        progress_callback=progress_callback,
    )


def resume_from_file(
    path: Path,
    settings: Settings,
    search_fn=None,
    progress_callback: Optional[Callable[[ResearchState], None]] = None,
) -> ResearchState:
    """Load a persisted state and continue the pipeline."""

    state = ResearchState.load(path)

    logger.info(
        "Resuming run '%s' from %s.",
        state.run_id,
        path,
    )

    return run_pipeline(
        state=state,
        settings=settings,
        search_fn=search_fn,
        resume=True,
        progress_callback=progress_callback,
    )


__all__ = [
    "run_pipeline",
    "run_new",
    "resume_from_file",
]