# Marketing Intelligence Agent

An AI agent that takes a business (e.g. "Toyota Surat") as input, researches
publicly available information about it and its market **without using any
paid/external research APIs**, and produces an evidence-backed marketing
strategy report using Claude as its reasoning engine.

Built for the AI/ML Junior Internship assignment at Vartalap AI, Surat.

> **Status: Phase 2 of 10 complete** (project setup / scaffolding).
> Research, Markdown knowledge layer, Claude reasoning, orchestration,
> validation, UI, and report generation land in subsequent phases — see
> [Implementation Phases](#implementation-phases) below.

---

## Problem

Build an agent that goes beyond generic chatbot marketing advice: given a
business, it should independently plan research, gather real public
information about the business/competitors/customers/content/market,
analyze it, and generate a specific, evidence-traceable marketing strategy.

**Hard constraints:**
- No external research/search APIs (Google Search API, SerpAPI, social APIs, etc.) — research is done via direct HTTP fetch + HTML parsing of publicly accessible pages, respecting `robots.txt`, rate limits, and ToS.
- Claude is the required reasoning/generation engine throughout.
- Claims in the final report must be traceable to actual collected evidence — no fabricated statistics, competitors, reviews, or sources.

## Architecture

A single orchestrated pipeline (not a multi-agent framework) with a shared,
serializable state object passed between stages:

```
BusinessInput
     │
     ▼
Research Planner (Claude)  ── decides which research areas & queries matter
     │
     ▼
Research Modules  ── business / competitor / customer / content / market
     │              (requests + BeautifulSoup; robots.txt-respecting)
     ▼
Markdown Knowledge Layer  ── python-frontmatter, one .md file per source
     │
     ▼
Extraction (Claude)  ── raw doc -> structured claims, evidence-checked
     │
     ▼
Analysis Modules (Claude)  ── 5 dimension-specific analyses over claims
     │
     ▼
Strategist (Claude)  ── positioning, campaigns, action plan (cites claim_ids)
     │
     ▼
Validator (Claude + code)  ── strips/flags any unsupported claim
     │
     ▼
Final Report (Markdown/HTML)
```

**Why this shape, not a multi-agent framework (LangGraph/CrewAI/AutoGen):**
the research dimensions are known upfront (business/competitor/customer/
content/market), so full autonomous re-planning isn't needed — a
deterministic, inspectable pipeline is easier to debug, easier to explain
in an interview, and easier to keep evidence-honest. The one genuinely
agentic decision point is the **Research Planner**, which uses Claude to
decide *what* to research and *which* queries to run given the specific
business/industry/location/objective — this is what makes the system work
for "Toyota Surat" and "a local restaurant" without hardcoded logic.

Full rationale for every technology choice (Claude model, `python-frontmatter`,
`requests`+`BeautifulSoup` over Selenium, Streamlit over alternatives, the
evidence-typing scheme) is in [`docs/architecture.md`](docs/architecture.md)
*(added when Phase 3 lands)*.

## Tech Stack

| Concern | Choice | Why (short) |
|---|---|---|
| LLM | Claude API (`anthropic` SDK), model `claude-sonnet-5` | Required by the assignment; no open-weight Claude exists, so API access is the only real option. Configurable via `.env`. |
| Web fetching | `requests` + `BeautifulSoup4` (+ `lxml`) | No API needed; standard, well-understood, respects robots.txt easily via `urllib.robotparser`. |
| Knowledge storage | Markdown + `python-frontmatter` | Human-readable, diff-able, keeps metadata (source/URL/confidence) attached to content. |
| State/schema validation | `pydantic` | Structured, serializable state object; validates Claude's JSON outputs before they flow downstream. |
| UI | Streamlit | Fastest path to a demoable, staged-progress UI within a 1-week budget. |
| Retries | `tenacity` | Clean retry/backoff for flaky network calls and LLM schema-validation retries. |

## Project Structure

```
marketing-intelligence-agent/
├── app/
│   ├── main.py            # Entry point (CLI for now; Streamlit calls into this later)
│   ├── config.py            # All settings loaded from .env, in one place
│   ├── agent/
│   │   └── state.py          # ResearchState — the object that flows through every stage
│   ├── research/            # Phase 3: business/competitor/customer/content/market fetchers
│   ├── processing/           # Phase 4: Markdown+frontmatter conversion, extraction helpers
│   ├── analysis/              # Phase 5: per-dimension Claude analysis modules
│   ├── strategy/                # Phase 5: strategist, campaign generator, action plan
│   └── validation/               # Phase 7: evidence/hallucination validator
├── data/
│   ├── raw/               # Unmodified fetched pages
│   ├── knowledge/          # Cleaned Markdown+frontmatter, organized by research dimension
│   └── reports/             # Final generated reports
├── tests/                  # pytest tests, one file per module as it's built
├── requirements.txt
├── .env.example
└── README.md   (this file)
```

## Setup

```bash
git clone <repo-url>
cd marketing-intelligence-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY to a real key from
# https://console.anthropic.com/
```

## Usage (current, Phase 2)

Right now `main.py` only proves the config → state → storage skeleton works;
no research or LLM calls happen yet.

```bash
python -m app.main \
  --business "Toyota Surat" \
  --industry "Automobile" \
  --location "Surat, Gujarat" \
  --objective "Increase customer acquisition"
```

This creates a `ResearchState`, logs the `init` stage, and saves it to
`data/state_toyota-surat-surat-gujarat.json`. Check the console output and
that file to confirm setup is correct before moving on.

Run tests:

```bash
pytest
```

## Implementation Phases

| Phase | Contents | Status |
|---|---|---|
| 1 | Architecture & design decisions | ✅ Done |
| 2 | Project scaffolding, config, state model, entry point | ✅ Done (this delivery) |
| 3 | Research layer (robots.txt-respecting fetchers per dimension) | ⏳ Next |
| 4 | Markdown knowledge layer (frontmatter, storage, retrieval) | Planned |
| 5 | Claude reasoning layer (extraction, per-dimension analysis, strategist) | Planned |
| 6 | Agent orchestration (wire all stages together) | Planned |
| 7 | Evidence/hallucination validation | Planned |
| 8 | Streamlit UI | Planned |
| 9 | Final report generation (Markdown/HTML) | Planned |
| 10 | Full documentation pass | Planned |

## Design Decisions Worth Knowing About

- **Competitor discovery is capped at `MAX_COMPETITORS` (default 6) for deep
  research**, even though discovery itself may surface more candidates.
  This is a deliberate scope decision to keep a live demo's runtime and
  token cost bounded — not a hidden limitation. Increase it in `.env` if
  you have time/budget for a bigger demo.
- **No automated bypass of robots.txt, CAPTCHAs, logins, or paywalls, ever.**
  Where public data isn't accessible, the system is designed to say so
  explicitly ("insufficient public data") rather than fabricate it — this is
  itself a defensible, honest output for a marketing intelligence report.
- **Every FACT/OBSERVATION claim must be traceable to a source-text excerpt.**
  INFERENCE/RECOMMENDATION claims must reference the claim_ids they were
  derived from. This is enforced in code, not just prompted.

## Limitations (current, will expand each phase)

- No research, analysis, or strategy generation is implemented yet — this
  delivery is scaffolding only.
- Customer/review research will be limited to genuinely public, non-gated
  pages; heavily-gated review platforms will legitimately show up as
  "insufficient public data" rather than being scraped around.

## Author

AI/ML Junior Internship candidate — Vartalap AI, Surat.
