"""
Shared web-fetching layer used by every research module (business,
competitor, customer, content, market — Phase 3 onward).

This is the ONLY place in the codebase that touches the network directly.
Every research module calls `fetch_url()` rather than using `requests`
itself, so the robots.txt check, rate limiting, and error handling only
need to be written once and are guaranteed to apply everywhere.

Design principles (from the assignment's "no APIs, respect access controls"
requirement):
  - robots.txt is checked before every single fetch, no exceptions.
  - Per-domain rate limiting is enforced (polite crawling).
  - Failures (timeout, 403, 404, connection error, non-HTML content) are
    caught and returned as a structured result — never raised uncaught,
    since one bad source should never crash a whole research run.
  - Nothing here ever attempts to bypass a login wall, CAPTCHA, or paywall.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.config import Settings

logger = logging.getLogger(__name__)

# Tags whose text is never part of the "main content" of a page.
_NOISE_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form")


@dataclass
class FetchResult:
    """
    The outcome of trying to fetch one URL. Always returned, never raised —
    callers check `.ok` rather than wrapping every call in try/except.
    """

    url: str
    ok: bool
    status_code: Optional[int] = None
    html: Optional[str] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    main_text: Optional[str] = None
    links: list[str] = field(default_factory=list)
    error: Optional[str] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_raw_document(self, category: str) -> dict:
        """
        Convert to the plain-dict shape stored in ResearchState.raw_documents
        (see app/agent/state.py). Kept as a dict there deliberately — see
        the note in state.py about not over-specifying schemas early.
        """
        return {
            "url": self.url,
            "category": category,
            "ok": self.ok,
            "status_code": self.status_code,
            "title": self.title,
            "meta_description": self.meta_description,
            "main_text": self.main_text,
            "links": self.links,
            "error": self.error,
            "fetched_at": self.fetched_at.isoformat(),
        }


class RobotsChecker:
    """
    Caches one urllib.robotparser.RobotFileParser per domain, so we don't
    re-fetch robots.txt on every single page request to the same site.
    """

    def __init__(self, user_agent: str):
        self._user_agent = user_agent
        self._parsers: dict[str, robotparser.RobotFileParser] = {}

    def _get_parser(self, domain_root: str) -> robotparser.RobotFileParser:
        if domain_root not in self._parsers:
            parser = robotparser.RobotFileParser()
            parser.set_url(urljoin(domain_root, "/robots.txt"))
            try:
                parser.read()
            except Exception as exc:  # noqa: BLE001 - robots.txt fetch failing is not fatal
                logger.warning("Could not read robots.txt for %s (%s) — assuming allowed.", domain_root, exc)
                # IMPORTANT: RobotFileParser.can_fetch() returns False by
                # default whenever `last_checked` was never set by a
                # successful read() — calling parse([]) alone does NOT fix
                # this, since parse() doesn't set last_checked either. That
                # would silently BLOCK every page whenever robots.txt is
                # unreachable, which is the opposite of the intended
                # behaviour (no stated restriction -> allow). Setting
                # allow_all directly short-circuits can_fetch() to True
                # regardless of last_checked.
                parser.allow_all = True
            self._parsers[domain_root] = parser
        return self._parsers[domain_root]

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        domain_root = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._get_parser(domain_root)
        return parser.can_fetch(self._user_agent, url)


class RateLimiter:
    """Enforces a minimum delay between requests to the same domain."""

    def __init__(self, delay_seconds: float):
        self._delay = delay_seconds
        self._last_request_at: dict[str, float] = {}

    def wait_if_needed(self, url: str) -> None:
        domain = urlparse(url).netloc
        last = self._last_request_at.get(domain)
        if last is not None:
            elapsed = time.monotonic() - last
            remaining = self._delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at[domain] = time.monotonic()


def extract_content(html: str, base_url: str) -> tuple[str, Optional[str], str, list[str]]:
    """
    Parse HTML into (title, meta_description, main_text, links).

    "Main text" is a simple but effective heuristic: strip script/style/nav/
    footer/header, then join remaining visible text. Good enough for
    marketing-research purposes (we mainly need paragraph/heading text, not
    pixel-perfect article extraction).
    """
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else None

    meta_description = None
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag["content"].strip()

    # Links must be collected BEFORE noise tags are stripped: nav/footer are
    # exactly where menu links usually live, so removing them first would
    # destroy the links we need (this was a real bug caught by the tests).
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        links.append(urljoin(base_url, href))

    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text_parts = [t.strip() for t in soup.stripped_strings]
    main_text = "\n".join(part for part in text_parts if part)

    return title, meta_description, main_text, links


class WebResearcher:
    """
    The single object every research module uses to fetch pages. Holds the
    shared robots-checker and rate-limiter so state (robots.txt cache,
    per-domain timing) is consistent across an entire run.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._robots = RobotsChecker(settings.user_agent)
        self._rate_limiter = RateLimiter(settings.request_delay_seconds)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": settings.user_agent})

    def fetch(self, url: str) -> FetchResult:
        """
        Fetch one URL, respecting robots.txt and rate limits. Never raises —
        every failure mode becomes a FetchResult with ok=False and a
        human-readable `error`.
        """
        if not self._robots.is_allowed(url):
            logger.info("Skipping %s — disallowed by robots.txt", url)
            return FetchResult(url=url, ok=False, error="Information unavailable — robots.txt disallows access")

        self._rate_limiter.wait_if_needed(url)

        try:
            response = self._session.get(url, timeout=self._settings.request_timeout_seconds)
        except requests.exceptions.Timeout:
            return FetchResult(url=url, ok=False, error="Information unavailable — request timed out")
        except requests.exceptions.ConnectionError as exc:
            return FetchResult(url=url, ok=False, error=f"Information unavailable — connection error ({exc})")
        except requests.exceptions.RequestException as exc:
            return FetchResult(url=url, ok=False, error=f"Information unavailable — request failed ({exc})")

        if response.status_code != 200:
            return FetchResult(
                url=url,
                ok=False,
                status_code=response.status_code,
                error=f"Information unavailable — HTTP {response.status_code}",
            )

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return FetchResult(
                url=url,
                ok=False,
                status_code=response.status_code,
                error=f"Information unavailable — non-HTML content ({content_type or 'unknown'})",
            )

        # requests falls back to ISO-8859-1 when a server's Content-Type
        # header doesn't declare a charset (common on older/misconfigured
        # sites). That default mis-decodes UTF-8 BOM bytes into garbage
        # characters (visible as "ï»¿" at the start of the extracted text).
        # response.apparent_encoding runs real content-sniffing (via
        # charset-normalizer) and is far more reliable here.
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding

        html_text = response.text.lstrip("\ufeff")  # strip a literal BOM if one survives decoding

        try:
            title, meta_description, main_text, links = extract_content(html_text, url)
        except Exception as exc:  # noqa: BLE001 - malformed HTML shouldn't crash the run
            return FetchResult(
                url=url,
                ok=False,
                status_code=response.status_code,
                error=f"Information unavailable — could not parse HTML ({exc})",
            )

        if not main_text.strip():
            return FetchResult(
                url=url,
                ok=False,
                status_code=response.status_code,
                error="Information unavailable — page had no extractable text",
            )

        return FetchResult(
            url=url,
            ok=True,
            status_code=response.status_code,
            html=html_text,
            title=title,
            meta_description=meta_description,
            main_text=main_text,
            links=links,
        )

    def fetch_many(self, urls: list[str], category: str) -> list[dict]:
        """
        Fetch a known list of URLs (no link-discovery) and tag them with
        the given category. Thin convenience wrapper around research_urls()
        for callers that already have a WebResearcher instance in hand.
        """
        return research_urls(self, urls, category)


# Keywords used to spot useful subpages among a homepage's nav links, e.g.
# distinguishing "About Us" and "Contact" from "Careers" or "Blog Post #42".
# Shared by business_research.py and competitor_research.py, since both
# fetch "a business's own website" — just for a different business.
USEFUL_SITE_PATH_KEYWORDS = (
    "about",
    "contact",
    "product",
    "service",
    "offer",
    "pricing",
    "location",
    "store",
    "dealership",
)

MAX_SITE_SUBPAGES = 4


def _same_domain(url: str, domain: str) -> bool:
    return urlparse(url).netloc == domain


def _pick_relevant_subpages(homepage_links: list[str], domain: str) -> list[str]:
    """
    From all links found on a homepage, pick a small set of same-domain
    links whose path looks relevant to business/competitor research, in the
    order they were found, capped and de-duplicated.
    """
    seen: set[str] = set()
    picked: list[str] = []
    for link in homepage_links:
        if not _same_domain(link, domain):
            continue
        path = urlparse(link).path.lower()
        if not any(keyword in path for keyword in USEFUL_SITE_PATH_KEYWORDS):
            continue
        normalized = link.split("#")[0].rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        picked.append(link)
        if len(picked) >= MAX_SITE_SUBPAGES:
            break
    return picked


def research_website(researcher: WebResearcher, homepage_url: str, category: str) -> list[dict]:
    """
    Fetch a business's (or competitor's) homepage plus a handful of
    relevant subpages discovered from the homepage's own navigation.

    Shared by business_research.py and competitor_research.py — the only
    difference between researching "the business" and "a competitor" is the
    `category` label attached to the resulting documents.

    Returns a list of raw_document dicts (see FetchResult.to_raw_document).
    Pages that fail to fetch are still included, with ok=False and a
    human-readable error — never silently dropped.
    """
    documents: list[dict] = []

    logger.info("Fetching %s homepage: %s", category, homepage_url)
    homepage_result = researcher.fetch(homepage_url)
    documents.append(homepage_result.to_raw_document(category=category))

    if not homepage_result.ok:
        logger.warning("Homepage fetch failed for %s: %s", homepage_url, homepage_result.error)
        return documents

    domain = urlparse(homepage_url).netloc
    subpages = _pick_relevant_subpages(homepage_result.links, domain)
    logger.info("Found %d candidate subpage(s) for %s: %s", len(subpages), homepage_url, subpages)

    for subpage_url in subpages:
        result = researcher.fetch(subpage_url)
        documents.append(result.to_raw_document(category=category))
        if not result.ok:
            logger.info("Subpage fetch failed for %s: %s", subpage_url, result.error)

    return documents


def research_urls(researcher: WebResearcher, urls: list[str], category: str) -> list[dict]:
    """
    Fetch an explicit list of URLs with no link-following — used for
    categories that don't have a single "homepage" to crawl from (customer
    reviews, content pieces, market/industry reports). The caller (the
    Research Planner, in Phase 5, or a user-supplied seed list) decides
    which URLs matter; this function just fetches them safely.
    """
    documents: list[dict] = []
    for url in urls:
        result = researcher.fetch(url)
        documents.append(result.to_raw_document(category=category))
        if not result.ok:
            logger.info("Fetch failed for %s (%s): %s", category, url, result.error)
    return documents