"""
Autonomous web discovery for the Marketing Intelligence Agent.

This module turns a business/category input into a structured research
source list.

The goal is NOT to let the LLM invent URLs. Instead, discovery is based on
search-engine result pages and deterministic URL classification.

The output can then be passed into the existing WebResearcher / extraction
pipeline.

Discovery categories:
    - business
    - competitor
    - campaign
    - content
    - customer
    - market

The module deliberately keeps discovery separate from:
    - web fetching
    - claim extraction
    - analysis
    - strategy generation

That separation makes each layer testable independently.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse

from app.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveryResult:
    """One URL discovered during research."""

    url: str
    title: str
    category: str
    query: str
    source: str = "web_search"


# ---------------------------------------------------------------------------
# Search result parsing
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+",
    re.IGNORECASE,
)

_TAG_RE = re.compile(r"<[^>]+>")

_SPACE_RE = re.compile(r"\s+")


def _clean_text(value: str) -> str:
    """Remove HTML noise and normalize whitespace."""
    value = unescape(value or "")
    value = _TAG_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def _normalize_url(url: str) -> str:
    """
    Normalize a URL enough for deduplication.

    Tracking parameters are removed where possible. Fragments are always
    removed because they do not represent a different page for research.
    """
    url = unescape(url).strip()

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return ""

    if not parsed.netloc:
        return ""

    query = parse_qs(parsed.query)

    # Remove common tracking parameters.
    tracking_prefixes = (
        "utm_",
        "fbclid",
        "gclid",
        "msclkid",
        "ref",
    )

    filtered = {}

    for key, values in query.items():
        key_lower = key.lower()

        if key_lower.startswith(tracking_prefixes):
            continue

        filtered[key] = values

    query_string = "&".join(
        f"{key}={value}"
        for key, values in filtered.items()
        for value in values
    )

    normalized = parsed._replace(
        fragment="",
        query=query_string,
    )

    result = normalized.geturl().rstrip("/")

    return result


def _is_search_engine_url(url: str) -> bool:
    """Prevent search result pages from becoming research documents."""
    host = urlparse(url).netloc.lower()

    blocked_hosts = {
        "google.com",
        "www.google.com",
        "bing.com",
        "www.bing.com",
        "search.yahoo.com",
        "duckduckgo.com",
        "www.duckduckgo.com",
    }

    return host in blocked_hosts


def _extract_urls_from_html(html: str) -> list[str]:
    """
    Extract HTTP(S) URLs from raw HTML.

    This intentionally uses a conservative parser rather than attempting
    to understand every search-engine DOM implementation.
    """
    if not html:
        return []

    urls: list[str] = []

    for match in _URL_RE.findall(html):
        cleaned = match.rstrip(").,;\"'")

        if cleaned:
            urls.append(cleaned)

    return urls


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def build_discovery_queries(
    business_name: str,
    industry: str,
    location: str,
) -> dict[str, list[str]]:
    """
    Build search queries for the major research dimensions.

    These queries are deliberately deterministic. The LLM does not decide
    which URLs exist.
    """

    business = business_name.strip()
    industry = industry.strip()
    location = location.strip()

    return {
        "business": [
            f'"{business}"',
            f'"{business}" {location}',
            f'"{business}" {industry}',
        ],
        "competitor": [
            f'"{business}" competitors',
            f'"{business}" alternatives',
            f'"{industry}" competitors {location}',
            f'"{industry}" brands {location}',
        ],
        "campaign": [
            f'"{business}" campaign',
            f'"{business}" marketing campaign',
            f'"{business}" promotion offer',
            f'"{business}" launch campaign',
        ],
        "content": [
            f'"{business}" blog',
            f'"{business}" content marketing',
            f'"{business}" articles',
            f'"{business}" social media',
        ],
        "customer": [
            f'"{business}" reviews',
            f'"{business}" customer reviews',
            f'"{business}" testimonials',
            f'"{business}" complaints',
        ],
        "market": [
            f'"{industry}" market trends',
            f'"{industry}" consumer trends',
            f'"{industry}" market report',
            f'"{industry}" {location} market',
        ],
    }


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

_CAMPAIGN_TERMS = (
    "campaign",
    "promotion",
    "offer",
    "sale",
    "launch",
    "event",
    "landing",
)

_CONTENT_TERMS = (
    "blog",
    "article",
    "news",
    "insights",
    "resources",
    "guide",
)

_CUSTOMER_TERMS = (
    "review",
    "reviews",
    "testimonial",
    "testimonials",
    "feedback",
    "complaint",
)

_MARKET_TERMS = (
    "market",
    "industry",
    "report",
    "research",
    "trend",
    "trends",
)


def classify_url(url: str, query_category: str) -> str:
    """
    Assign a research category to a discovered URL.

    Query category is the fallback. URL semantics refine it where possible.
    """
    lowered = url.lower()
    path = urlparse(url).path.lower()

    if query_category == "campaign":
        if any(term in lowered for term in _CAMPAIGN_TERMS):
            return "campaign"
        return "campaign"

    if query_category == "content":
        if any(term in path for term in _CONTENT_TERMS):
            return "content"
        return "content"

    if query_category == "customer":
        if any(term in lowered for term in _CUSTOMER_TERMS):
            return "customer"
        return "customer"

    if query_category == "market":
        if any(term in lowered for term in _MARKET_TERMS):
            return "market"
        return "market"

    return query_category


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------

class WebDiscovery:
    """
    Search-driven source discovery.

    `search_fn` is injected so tests can mock search-engine access and the
    application can later plug in Google/Bing/Brave/SerpAPI/etc. without
    changing the discovery logic.

    Expected search_fn contract:

        search_fn(query: str, max_results: int) -> list[dict]

    Each result may contain:

        {
            "url": "...",
            "title": "..."
        }
    """

    def __init__(
        self,
        settings: Settings,
        search_fn=None,
    ) -> None:
        self.settings = settings
        self.search_fn = search_fn

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[DiscoveryResult]:
        """Execute one injected search and normalize its results."""

        if self.search_fn is None:
            logger.warning(
                "No search provider configured. Query skipped: %s",
                query,
            )
            return []

        try:
            raw_results = self.search_fn(
                query=query,
                max_results=max_results,
            )
        except TypeError:
            # Also support a simple positional callable.
            raw_results = self.search_fn(query, max_results)

        if not raw_results:
            return []

        results: list[DiscoveryResult] = []
        seen: set[str] = set()

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            url = _normalize_url(str(item.get("url", "")))

            if not url:
                continue

            if _is_search_engine_url(url):
                continue

            if url in seen:
                continue

            seen.add(url)

            results.append(
                DiscoveryResult(
                    url=url,
                    title=_clean_text(str(item.get("title", ""))),
                    category="",
                    query=query,
                )
            )

        return results

    def discover_category(
        self,
        category: str,
        queries: Iterable[str],
        max_results_per_query: int = 10,
        max_total: int = 20,
    ) -> list[DiscoveryResult]:
        """Discover and deduplicate sources for one research category."""

        discovered: list[DiscoveryResult] = []
        seen: set[str] = set()

        for query in queries:
            results = self.search(
                query=query,
                max_results=max_results_per_query,
            )

            for result in results:
                if result.url in seen:
                    continue

                seen.add(result.url)

                discovered.append(
                    DiscoveryResult(
                        url=result.url,
                        title=result.title,
                        category=classify_url(result.url, category),
                        query=query,
                        source=result.source,
                    )
                )

                if len(discovered) >= max_total:
                    return discovered

        return discovered

    def discover(
        self,
        business_name: str,
        industry: str,
        location: str,
        max_results_per_query: int = 10,
    ) -> dict[str, list[dict]]:
        """
        Run discovery across all research dimensions.

        Returns plain dictionaries so the result can be stored directly
        inside ResearchState / JSON.
        """

        queries = build_discovery_queries(
            business_name=business_name,
            industry=industry,
            location=location,
        )

        output: dict[str, list[dict]] = {}

        # Keep the overall research bounded.
        limits = {
            "business": 10,
            "competitor": self.settings.max_competitors * 4,
            "campaign": 12,
            "content": 10,
            "customer": 10,
            "market": 10,
        }

        for category, category_queries in queries.items():
            results = self.discover_category(
                category=category,
                queries=category_queries,
                max_results_per_query=max_results_per_query,
                max_total=limits[category],
            )

            output[category] = [
                {
                    "url": result.url,
                    "title": result.title,
                    "category": result.category,
                    "query": result.query,
                    "source": result.source,
                }
                for result in results
            ]

            logger.info(
                "Discovery category '%s': %d source(s)",
                category,
                len(results),
            )

        return output


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def discover_research_sources(
    business_name: str,
    industry: str,
    location: str,
    settings: Settings,
    search_fn=None,
) -> dict[str, list[dict]]:
    """
    Convenience wrapper used by the future orchestrator.
    """

    discovery = WebDiscovery(
        settings=settings,
        search_fn=search_fn,
    )

    return discovery.discover(
        business_name=business_name,
        industry=industry,
        location=location,
    )


__all__ = [
    "DiscoveryResult",
    "WebDiscovery",
    "build_discovery_queries",
    "classify_url",
    "discover_research_sources",
]