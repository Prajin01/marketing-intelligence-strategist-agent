"""
Live demo script for Phase 3 — NOT part of the test suite (it makes real
network requests, so it can't be relied on to pass/fail deterministically
in CI). Use this to see the research layer actually fetch a real website
and print what it found.

Usage:
    python -m scripts.demo_research https://www.example.com
    python -m scripts.demo_research https://www.example.com --dimension competitor

Respects robots.txt exactly like the real pipeline will — if the site you
point it at disallows scraping, you'll see that reported honestly rather
than bypassed.
"""

from __future__ import annotations

import argparse
import sys

from app.config import configure_logging, load_settings
from app.processing.markdown_processor import save_raw_documents
from app.research.business_research import research_business
from app.research.competitor_research import research_competitor
from app.research.web_research import WebResearcher

_DIMENSION_FUNCS = {
    "business": research_business,
    "competitor": research_competitor,
}


def _print_document(doc: dict, index: int) -> None:
    print(f"\n--- Document {index} ---")
    print(f"URL:      {doc['url']}")
    print(f"Category: {doc['category']}")
    print(f"OK:       {doc['ok']}")
    if doc["ok"]:
        print(f"Title:    {doc['title']}")
        print(f"Meta:     {doc['meta_description']}")
        snippet = (doc["main_text"] or "")[:300].replace("\n", " ")
        print(f"Text (first 300 chars): {snippet}...")
    else:
        print(f"Error:    {doc['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live demo of the Phase 3 research layer")
    parser.add_argument("homepage_url", help="A real homepage URL to research, e.g. https://example.com")
    parser.add_argument(
        "--dimension",
        choices=list(_DIMENSION_FUNCS.keys()),
        default="business",
        help="Which researcher to run (default: business)",
    )
    parser.add_argument(
        "--business-name",
        default="Demo Business",
        help="Business name to tag in the saved knowledge documents' frontmatter",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also save fetched pages as real .md knowledge files under data/knowledge/ (Phase 4)",
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_level)

    researcher = WebResearcher(settings)
    research_fn = _DIMENSION_FUNCS[args.dimension]

    print(f"Researching ({args.dimension}): {args.homepage_url}")
    print(f"User-Agent: {settings.user_agent}")
    print(f"Rate limit delay: {settings.request_delay_seconds}s between requests to the same domain\n")

    documents = research_fn(researcher, args.homepage_url)

    print(f"\n=== Fetched {len(documents)} document(s) ===")
    for i, doc in enumerate(documents, start=1):
        _print_document(doc, i)

    ok_count = sum(1 for d in documents if d["ok"])
    print(f"\n=== Summary: {ok_count}/{len(documents)} pages fetched successfully ===")

    if args.save:
        saved_paths = save_raw_documents(documents, business_name=args.business_name, settings=settings)
        print(f"\n=== Saved {len(saved_paths)} knowledge document(s) ===")
        for path in saved_paths:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())