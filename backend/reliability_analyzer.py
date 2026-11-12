
"""
Source Reliability Analyzer

Evaluates the reliability of retrieved evidence sources.

Features:
1. Domain extraction and normalization
2. Recognized source registry
3. Source category assessment
4. HTTPS assessment
5. Publisher identification
6. URL transparency assessment
7. Weighted reliability scoring
8. Reliability classification
9. Explainable reliability reports

Important:
This module evaluates source-level reliability signals.
It does not determine whether an individual claim is true.
"""

from urllib.parse import urlparse

from backend.data_contracts import (
    Evidence,
    SourceReliability
)


# ==========================================
# CONFIGURATION
# ==========================================

RELIABILITY_WEIGHTS = {
    "domain_reputation": 0.30,
    "source_category": 0.25,
    "https": 0.10,
    "publisher_identification": 0.20,
    "url_transparency": 0.15
}


RELIABILITY_THRESHOLDS = {
    "HIGH": 0.80,
    "MEDIUM": 0.60
}


# ==========================================
# RECOGNIZED SOURCE REGISTRY
# ==========================================

# Initial heuristic registry values.
# These are not empirically validated credibility ratings.

TRUSTED_SOURCES = {

    "nasa.gov": {
        "domain_reputation": 0.95,
        "source_category": 0.95,
        "publisher_identification": 0.95
    },

    "who.int": {
        "domain_reputation": 0.95,
        "source_category": 0.95,
        "publisher_identification": 0.95
    },

    "gov.in": {
        "domain_reputation": 0.90,
        "source_category": 0.90,
        "publisher_identification": 0.90
    },

    "nic.in": {
        "domain_reputation": 0.90,
        "source_category": 0.90,
        "publisher_identification": 0.90
    },

    "edu": {
        "domain_reputation": 0.85,
        "source_category": 0.90,
        "publisher_identification": 0.85
    },

    "wikipedia.org": {
        "domain_reputation": 0.80,
        "source_category": 0.80,
        "publisher_identification": 0.85
    }
}


# ==========================================
# SOURCE CATEGORY SCORES
# ==========================================

SOURCE_CATEGORY_SCORES = {

    "government": 0.95,
    "academic": 0.90,
    "encyclopedia": 0.80,
    "research": 0.90,
    "news": 0.75,
    "webpage": 0.50,
    "unknown": 0.40
}


# ==========================================
# DOMAIN UTILITIES
# ==========================================

def extract_domain(url):
    """
    Extract and normalize the hostname from a URL.

    Returns:
        Normalized domain name.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    try:
        parsed_url = urlparse(url.strip())

        domain = parsed_url.hostname

        if not domain:
            raise ValueError("URL does not contain a valid hostname.")

        domain = domain.lower().rstrip(".")

        return domain

    except ValueError:
        raise ValueError("Invalid URL.")


def is_valid_url(url):
    """
    Perform basic URL structure validation.

    This does not verify whether the website exists.
    """

    if not isinstance(url, str) or not url.strip():
        return False

    try:
        parsed_url = urlparse(url.strip())

        if parsed_url.scheme not in {"http", "https"}:
            return False

        if not parsed_url.hostname or "." not in parsed_url.hostname:
            return False

        if " " in url:
            return False

        if parsed_url.port is not None:
            if not 1 <= parsed_url.port <= 65535:
                return False

        return True

    except ValueError:
        return False


# ==========================================
# DOMAIN REGISTRY LOOKUP
# ==========================================

def get_registered_source(domain):
    """
    Identify a recognized source using exact domain
    or controlled subdomain matching.
    """

    if not isinstance(domain, str) or not domain.strip():
        return None

    domain = domain.lower().strip().rstrip(".")

    # Exact domain match.
    if domain in TRUSTED_SOURCES:
        return TRUSTED_SOURCES[domain]

    # Match known parent domains.
    for registered_domain, details in TRUSTED_SOURCES.items():

        if registered_domain == "edu":
            if domain.endswith(".edu") or ".edu." in domain:
                return details

        elif domain.endswith("." + registered_domain):
            return details

    return None


# ==========================================
# DOMAIN REPUTATION ANALYSIS
# ==========================================

def assess_domain_reputation(domain):
    """
    Evaluate domain reputation using the local registry.

    Unknown domains receive a cautious neutral score.
    """

    registered_source = get_registered_source(domain)

    if registered_source:
        return (
            registered_source["domain_reputation"],
            "Domain matches a recognized source registry entry."
        )

    return (
        0.50,
        "Domain is not present in the local source registry."
    )


# ==========================================
# SOURCE CATEGORY ANALYSIS
# ==========================================

def assess_source_category(source_type, domain):
    """
    Evaluate source category.

    Uses the existing Evidence.source_type field
    and recognized domain information.
    """

    if not isinstance(source_type, str):
        raise ValueError("Source type must be a string.")

    normalized_type = source_type.lower().strip()

    registered_source = get_registered_source(domain)

    if registered_source:

        if domain == "wikipedia.org" or domain.endswith(".wikipedia.org"):
            return (
                SOURCE_CATEGORY_SCORES["encyclopedia"],
                "Recognized encyclopedia domain."
            )

        if domain.endswith(".gov") or domain.endswith(".gov.in"):
            return (
                SOURCE_CATEGORY_SCORES["government"],
                "Recognized government domain."
            )

        if domain.endswith(".edu") or domain.endswith(".edu.in"):
            return (
                SOURCE_CATEGORY_SCORES["academic"],
                "Recognized academic domain."
            )

        return (
            registered_source["source_category"],
            "Source category obtained from the local registry."
        )

    score = SOURCE_CATEGORY_SCORES.get(
        normalized_type,
        SOURCE_CATEGORY_SCORES["unknown"]
    )

    return (
        score,
        f"Source category identified as '{normalized_type}'."
    )


# ==========================================
# HTTPS ANALYSIS
# ==========================================

def assess_https(url):
    """
    Check whether the URL uses HTTPS.

    HTTPS only indicates encrypted transport.
    It does not guarantee source credibility.
    """

    parsed_url = urlparse(url.strip())

    if parsed_url.scheme.lower() == "https":

        return (
            1.0,
            "Source uses HTTPS encrypted transport."
        )

    return (
        0.0,
        "Source does not use HTTPS."
    )


# ==========================================
# PUBLISHER IDENTIFICATION
# ==========================================

def assess_publisher_identification(source_title, domain):
    """
    Estimate publisher identification from available
    source title and domain information.

    No external publisher verification is performed.
    """

    if not source_title or not source_title.strip():

        return (
            0.30,
            "Source title is unavailable."
        )

    registered_source = get_registered_source(domain)

    if registered_source:

        return (
            registered_source["publisher_identification"],
            "Publisher is associated with a recognized registry entry."
        )

    if domain:

        return (
            0.60,
            "Source title and hostname are available, but publisher identity is not independently verified."
        )

    return (
        0.30,
        "Publisher identity cannot be assessed."
    )


# ==========================================
# URL TRANSPARENCY
# ==========================================

def assess_url_transparency(url):
    """
    Evaluate basic URL transparency.

    Checks:
    - Valid URL structure
    - Hostname availability
    - Reasonable path structure
    """

    if not is_valid_url(url):

        return (
            0.0,
            "URL structure is invalid."
        )

    parsed_url = urlparse(url.strip())

    domain = parsed_url.hostname

    if not domain:

        return (
            0.0,
            "URL hostname is unavailable."
        )

    if parsed_url.path in {"", "/"}:

        return (
            0.70,
            "URL has a valid hostname but no specific resource path."
        )

    if len(parsed_url.path) > 200:

        return (
            0.60,
            "URL has an unusually long resource path."
        )

    return (
        0.90,
        "URL contains a valid hostname and identifiable resource path."
    )


# ==========================================
# WEIGHTED RELIABILITY SCORE
# ==========================================

def calculate_reliability_score(reliability_factors):
    """
    Calculate weighted source reliability.

    Formula:

    Score =
        (Domain Reputation * 0.30)
        + (Source Category * 0.25)
        + (HTTPS * 0.10)
        + (Publisher Identification * 0.20)
        + (URL Transparency * 0.15)
    """

    if not isinstance(reliability_factors, dict):
        raise ValueError("Reliability factors must be a dictionary.")

    if set(reliability_factors.keys()) != set(RELIABILITY_WEIGHTS.keys()):
        raise ValueError("Missing or invalid reliability factors.")

    total_score = 0.0

    for factor, weight in RELIABILITY_WEIGHTS.items():

        score = reliability_factors[factor]

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(
                f"Invalid reliability score for {factor}."
            )

        total_score += score * weight

    return round(total_score, 4)


# ==========================================
# RELIABILITY CLASSIFICATION
# ==========================================

def classify_reliability(score):
    """
    Convert reliability score into a category.
    """

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError("Score must be between 0 and 1.")

    if score >= RELIABILITY_THRESHOLDS["HIGH"]:
        return "HIGH"

    if score >= RELIABILITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"

    return "LOW"


# ==========================================
# EXPLANATION GENERATION
# ==========================================

def generate_reliability_explanation(
    reliability_factors,
    factor_explanations,
    category,
    score
):
    """
    Generate a human-readable reliability explanation.
    """

    explanation_parts = []

    explanation_parts.append(
        f"Source reliability is classified as {category} "
        f"with a score of {score:.2f}."
    )

    for factor, factor_score in reliability_factors.items():

        reason = factor_explanations[factor]

        explanation_parts.append(
            f"{factor.replace('_', ' ').title()}: "
            f"{factor_score:.2f}. {reason}"
        )

    explanation_parts.append(
        "This is a heuristic source-level assessment, "
        "not a guarantee of factual accuracy."
    )

    return " ".join(explanation_parts)


# ==========================================
# MAIN SOURCE RELIABILITY ANALYZER
# ==========================================

def analyze_source_reliability(evidence):
    """
    Analyze the reliability of a single Evidence object.

    Args:
        evidence: Evidence object.

    Returns:
        SourceReliability object.
    """

    if not isinstance(evidence, Evidence):
        raise ValueError("Input must be an Evidence object.")

    if not is_valid_url(evidence.source_url):
        raise ValueError("Evidence contains an invalid source URL.")

    domain = extract_domain(evidence.source_url)

    # Analyze individual reliability factors.

    domain_score, domain_reason = assess_domain_reputation(
        domain
    )

    category_score, category_reason = assess_source_category(
        evidence.source_type,
        domain
    )

    https_score, https_reason = assess_https(
        evidence.source_url
    )

    publisher_score, publisher_reason = assess_publisher_identification(
        evidence.source_title,
        domain
    )

    transparency_score, transparency_reason = assess_url_transparency(
        evidence.source_url
    )

    # Collect factor scores.

    reliability_factors = {
        "domain_reputation": domain_score,
        "source_category": category_score,
        "https": https_score,
        "publisher_identification": publisher_score,
        "url_transparency": transparency_score
    }

    # Collect explanations.

    factor_explanations = {
        "domain_reputation": domain_reason,
        "source_category": category_reason,
        "https": https_reason,
        "publisher_identification": publisher_reason,
        "url_transparency": transparency_reason
    }

    # Calculate overall score.

    reliability_score = calculate_reliability_score(
        reliability_factors
    )

    # Classify source.

    reliability_category = classify_reliability(
        reliability_score
    )

    # Generate explanation.

    explanation = generate_reliability_explanation(
        reliability_factors,
        factor_explanations,
        reliability_category,
        reliability_score
    )

    # Return structured contract.

    return SourceReliability(
        evidence_id=evidence.evidence_id,
        reliability_score=reliability_score,
        reliability_category=reliability_category,
        reliability_factors=reliability_factors,
        explanation=explanation
    )


# ==========================================
# BATCH SOURCE RELIABILITY ANALYSIS
# ==========================================

def analyze_sources(evidence_list):
    """
    Analyze reliability for multiple Evidence objects.

    Returns:
        List of SourceReliability objects.
    """

    if not isinstance(evidence_list, list):
        raise ValueError("Evidence input must be a list.")

    results = []

    for evidence in evidence_list:

        if not isinstance(evidence, Evidence):
            raise ValueError(
                "Every item must be an Evidence object."
            )

        results.append(
            analyze_source_reliability(evidence)
        )

    return results
