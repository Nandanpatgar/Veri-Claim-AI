
import re
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from tavily import TavilyClient

from backend.config import (
    WIKIPEDIA_API_URL,
    MAX_WIKIPEDIA_RESULTS,
    MAX_TAVILY_RESULTS,
    MAX_EVIDENCE_PER_CLAIM,
    TAVILY_API_KEY
)

from backend.data_contracts import Evidence


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "that", "this", "it", "its", "as",
    "be", "has", "have", "had", "which", "who", "what",
    "when", "where", "how"
}


# --------------------------------------------------
# TEXT RELEVANCE
# --------------------------------------------------

def tokenize(text):
    """Convert text into meaningful lowercase words."""

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def is_relevant(claim_text, evidence_text):
    """
    Check lexical relevance between a claim and evidence.

    This is only an initial filtering mechanism.
    It does not determine factual correctness.
    """

    claim_terms = tokenize(claim_text)
    evidence_terms = tokenize(evidence_text)

    if not claim_terms or not evidence_terms:
        return False

    matching_terms = claim_terms.intersection(evidence_terms)

    relevance_score = len(matching_terms) / len(claim_terms)

    return relevance_score >= 0.20


# --------------------------------------------------
# WIKIPEDIA SEARCH
# --------------------------------------------------

def search_wikipedia(claim_text):
    """Search Wikipedia for articles related to a claim."""

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": claim_text,
        "srlimit": MAX_WIKIPEDIA_RESULTS
    }

    headers = {
        "User-Agent": "Veri-Claim-AI/2.0 (evidence research)"
    }

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        return data.get("query", {}).get("search", [])

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Wikipedia search failed: {error}"
        ) from error


# --------------------------------------------------
# WIKIPEDIA ARTICLE EXTRACTION
# --------------------------------------------------

def fetch_wikipedia_extracts(results):
    """
    Fetch actual plain-text article extracts.

    Returns a dictionary containing page IDs and extracts.
    """

    page_ids = [
        str(result["pageid"])
        for result in results
        if result.get("pageid")
    ]

    if not page_ids:
        return {}

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exchars": 2500,
        "pageids": "|".join(page_ids)
    }

    headers = {
        "User-Agent": "Veri-Claim-AI/2.0 (evidence research)"
    }

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        pages = data.get("query", {}).get("pages", {})

        extracts = {}

        for page_id, page_data in pages.items():

            extract = page_data.get("extract", "").strip()

            if extract:
                extracts[str(page_id)] = extract

        return extracts

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Wikipedia article extraction failed: {error}"
        ) from error


def extract_wikipedia_evidence(
    claim_id,
    claim_text,
    results,
    extracts
):
    """Convert relevant Wikipedia extracts into Evidence objects."""

    evidence_list = []

    for result in results:

        title = result.get("title", "").strip()
        page_id = result.get("pageid")

        if not title or not page_id:
            continue

        passage = extracts.get(str(page_id), "").strip()

        if not passage:
            continue

        combined_text = f"{title} {passage}"

        if not is_relevant(claim_text, combined_text):
            continue

        article_title = quote(
            title.replace(" ", "_"),
            safe="()_,-"
        )

        evidence = Evidence(
            claim_id=claim_id,
            passage=passage,
            source_title=title,
            source_url=(
                f"https://en.wikipedia.org/wiki/{article_title}"
            ),
            source_type="encyclopedia",
            retrieval_provider="Wikipedia"
        )

        evidence_list.append(evidence)

    return evidence_list


# --------------------------------------------------
# TAVILY SEARCH
# --------------------------------------------------

def search_tavily(claim_text):
    """Search the web using Tavily."""

    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is missing from the environment."
        )

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=claim_text,
            search_depth="advanced",
            max_results=MAX_TAVILY_RESULTS,
            include_answer=False
        )

        return response.get("results", [])

    except Exception as error:
        raise RuntimeError(
            f"Tavily search failed: {error}"
        ) from error


def extract_tavily_evidence(
    claim_id,
    claim_text,
    results
):
    """Convert relevant Tavily results into Evidence objects."""

    evidence_list = []

    for result in results:

        title = result.get("title", "").strip()
        passage = result.get("content", "").strip()
        source_url = result.get("url", "").strip()

        if not title or not passage or not source_url:
            continue

        combined_text = f"{title} {passage}"

        if not is_relevant(claim_text, combined_text):
            continue

        evidence = Evidence(
            claim_id=claim_id,
            passage=passage,
            source_title=title,
            source_url=source_url,
            source_type="webpage",
            retrieval_provider="Tavily"
        )

        evidence_list.append(evidence)

    return evidence_list


# --------------------------------------------------
# URL NORMALIZATION
# --------------------------------------------------

def normalize_url(url):
    """Normalize URLs for duplicate detection."""

    try:
        parsed = urlsplit(url.strip())

        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        port = parsed.port

        netloc = hostname

        if port and not (
            (scheme == "https" and port == 443)
            or (scheme == "http" and port == 80)
        ):
            netloc = f"{hostname}:{port}"

        path = parsed.path.rstrip("/")

        return urlunsplit((
            scheme,
            netloc,
            path,
            "",
            ""
        )).lower()

    except (ValueError, AttributeError):
        return url.rstrip("/").lower()


# --------------------------------------------------
# EVIDENCE DEDUPLICATION
# --------------------------------------------------

def deduplicate_evidence(evidence_list):
    """Remove duplicate URLs across retrieval providers."""

    unique_evidence = []
    seen_urls = set()

    for evidence in evidence_list:

        normalized_url = normalize_url(evidence.source_url)

        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)

        unique_evidence.append(evidence)

    return unique_evidence


# --------------------------------------------------
# RETRIEVE EVIDENCE FOR ONE CLAIM
# --------------------------------------------------

def retrieve_evidence_for_claim(claim):
    """Retrieve evidence using Wikipedia and Tavily."""

    evidence_list = []
    errors = {}

    provider_status = {
        "Wikipedia": {
            "status": "pending",
            "result_count": 0,
            "error": None
        },
        "Tavily": {
            "status": "pending",
            "result_count": 0,
            "error": None
        }
    }

    # WIKIPEDIA

    try:

        wikipedia_results = search_wikipedia(
            claim.claim_text
        )

        wikipedia_extracts = fetch_wikipedia_extracts(
            wikipedia_results
        )

        wikipedia_evidence = extract_wikipedia_evidence(
            claim.claim_id,
            claim.claim_text,
            wikipedia_results,
            wikipedia_extracts
        )

        evidence_list.extend(wikipedia_evidence)

        provider_status["Wikipedia"]["status"] = "success"

        provider_status["Wikipedia"]["result_count"] = len(
            wikipedia_evidence
        )

    except Exception as error:

        errors["Wikipedia"] = str(error)

        provider_status["Wikipedia"]["status"] = "failed"

        provider_status["Wikipedia"]["error"] = str(error)

    # TAVILY

    try:

        tavily_results = search_tavily(
            claim.claim_text
        )

        tavily_evidence = extract_tavily_evidence(
            claim.claim_id,
            claim.claim_text,
            tavily_results
        )

        evidence_list.extend(tavily_evidence)

        provider_status["Tavily"]["status"] = "success"

        provider_status["Tavily"]["result_count"] = len(
            tavily_evidence
        )

    except Exception as error:

        errors["Tavily"] = str(error)

        provider_status["Tavily"]["status"] = "failed"

        provider_status["Tavily"]["error"] = str(error)

    # DEDUPLICATION

    evidence_list = deduplicate_evidence(evidence_list)

    # LIMIT RESULTS

    evidence_list = evidence_list[:MAX_EVIDENCE_PER_CLAIM]

    return {
        "claim_id": claim.claim_id,
        "evidence": evidence_list,
        "errors": errors,
        "provider_status": provider_status
    }


# --------------------------------------------------
# RETRIEVE EVIDENCE FOR MULTIPLE CLAIMS
# --------------------------------------------------

def retrieve_evidence(claims):
    """Retrieve evidence for a list of Claim objects."""

    if not isinstance(claims, list):

        raise ValueError(
            "Claims must be provided as a list."
        )

    all_evidence = []
    retrieval_report = {}

    for claim in claims:

        if not hasattr(claim, "claim_id") or not hasattr(
            claim, "claim_text"
        ):

            raise ValueError(
                "Every item must be a valid Claim object."
            )

        result = retrieve_evidence_for_claim(claim)

        all_evidence.extend(result["evidence"])

        retrieval_report[claim.claim_id] = {
            "evidence_count": len(result["evidence"]),
            "errors": result.get("errors", {}),
            "provider_status": result.get(
                "provider_status", {}
            )
        }

    return {
        "evidence": all_evidence,
        "retrieval_report": retrieval_report
    }
