"""
Unit tests for the Source Reliability Analyzer.

Tests:
1. URL validation
2. Domain extraction
3. Source registry
4. Reliability factors
5. Weighted scoring
6. Reliability classification
7. Single evidence analysis
8. Batch evidence analysis
9. Invalid inputs
"""

import math

import pytest

from backend.data_contracts import Evidence, SourceReliability

from backend.reliability_analyzer import (
    RELIABILITY_WEIGHTS,
    extract_domain,
    is_valid_url,
    get_registered_source,
    assess_domain_reputation,
    assess_source_category,
    assess_https,
    assess_publisher_identification,
    assess_url_transparency,
    calculate_reliability_score,
    classify_reliability,
    analyze_source_reliability,
    analyze_sources,
)


# ==========================================
# TEST FIXTURES
# ==========================================

@pytest.fixture
def nasa_evidence():
    return Evidence(
        claim_id="claim-001",
        passage="NASA studies Earth and space.",
        source_title="NASA Science",
        source_url="https://science.nasa.gov/earth/",
        source_type="government",
        retrieval_provider="Tavily",
        evidence_id="evidence-001",
    )


@pytest.fixture
def wikipedia_evidence():
    return Evidence(
        claim_id="claim-001",
        passage="India gained independence in 1947.",
        source_title="Independence Day (India)",
        source_url="https://en.wikipedia.org/wiki/Independence_Day_(India)",
        source_type="encyclopedia",
        retrieval_provider="Wikipedia",
        evidence_id="evidence-002",
    )


@pytest.fixture
def unknown_evidence():
    return Evidence(
        claim_id="claim-001",
        passage="An example statement.",
        source_title="Example Article",
        source_url="https://example.com/article",
        source_type="webpage",
        retrieval_provider="Tavily",
        evidence_id="evidence-003",
    )


@pytest.fixture
def valid_factors():
    return {
        "domain_reputation": 0.90,
        "source_category": 0.90,
        "https": 1.00,
        "publisher_identification": 0.90,
        "url_transparency": 0.90,
    }


# ==========================================
# URL AND DOMAIN TESTS
# ==========================================

def test_extract_domain():
    assert extract_domain("https://science.nasa.gov/earth/") == "science.nasa.gov"


def test_extract_domain_normalization():
    assert extract_domain("https://WWW.NASA.GOV/path") == "www.nasa.gov"


def test_extract_domain_invalid_input():
    with pytest.raises(ValueError):
        extract_domain("")


@pytest.mark.parametrize(
    "url",
    [
        "https://nasa.gov",
        "http://example.com",
        "https://en.wikipedia.org/wiki/India",
    ],
)
def test_valid_urls(url):
    assert is_valid_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "ftp://example.com",
        "https://",
        "https://example.com:99999",
    ],
)
def test_invalid_urls(url):
    assert is_valid_url(url) is False


# ==========================================
# SOURCE REGISTRY TESTS
# ==========================================

def test_registered_source_exact_match():
    assert get_registered_source("nasa.gov") is not None


def test_registered_source_subdomain():
    assert get_registered_source("science.nasa.gov") is not None


def test_unknown_source():
    assert get_registered_source("example.com") is None


def test_registry_rejects_similar_domain():
    assert get_registered_source("fake-nasa.gov") is None


# ==========================================
# RELIABILITY FACTOR TESTS
# ==========================================

def test_known_domain_reputation():
    score, explanation = assess_domain_reputation("nasa.gov")

    assert score == 0.95
    assert explanation


def test_unknown_domain_reputation():
    score, explanation = assess_domain_reputation("example.com")

    assert score == 0.50
    assert explanation


def test_wikipedia_category():
    score, explanation = assess_source_category(
        "encyclopedia",
        "en.wikipedia.org",
    )

    assert score == 0.80
    assert explanation


def test_unknown_webpage_category():
    score, explanation = assess_source_category(
        "webpage",
        "example.com",
    )

    assert score == 0.50
    assert explanation


def test_https_source():
    score, explanation = assess_https("https://example.com")

    assert score == 1.0
    assert explanation


def test_http_source():
    score, explanation = assess_https("http://example.com")

    assert score == 0.0
    assert explanation


def test_registered_publisher():
    score, explanation = assess_publisher_identification(
        "NASA Science",
        "nasa.gov",
    )

    assert score == 0.95
    assert explanation


def test_unknown_publisher():
    score, explanation = assess_publisher_identification(
        "Example Article",
        "example.com",
    )

    assert score == 0.60
    assert explanation


def test_url_transparency_with_path():
    score, explanation = assess_url_transparency(
        "https://example.com/article"
    )

    assert score == 0.90
    assert explanation


def test_url_transparency_without_path():
    score, explanation = assess_url_transparency(
        "https://example.com"
    )

    assert score == 0.70
    assert explanation


def test_invalid_url_transparency():
    score, explanation = assess_url_transparency("invalid-url")

    assert score == 0.0
    assert explanation


# ==========================================
# WEIGHTED SCORING TESTS
# ==========================================

def test_weights_sum_to_one():
    assert sum(RELIABILITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_calculate_reliability_score(valid_factors):
    score = calculate_reliability_score(valid_factors)

    expected = (
        0.90 * 0.30
        + 0.90 * 0.25
        + 1.00 * 0.10
        + 0.90 * 0.20
        + 0.90 * 0.15
    )

    assert score == pytest.approx(expected)


def test_reliability_score_missing_factor(valid_factors):
    valid_factors.pop("https")

    with pytest.raises(ValueError):
        calculate_reliability_score(valid_factors)


@pytest.mark.parametrize(
    "invalid_score",
    [
        -0.1,
        1.1,
        True,
        math.nan,
        math.inf,
    ],
)
def test_reliability_score_invalid_factor(valid_factors, invalid_score):
    valid_factors["https"] = invalid_score

    with pytest.raises(ValueError):
        calculate_reliability_score(valid_factors)


# ==========================================
# CLASSIFICATION TESTS
# ==========================================

@pytest.mark.parametrize(
    "score, expected",
    [
        (1.0, "HIGH"),
        (0.80, "HIGH"),
        (0.79, "MEDIUM"),
        (0.60, "MEDIUM"),
        (0.59, "LOW"),
        (0.0, "LOW"),
    ],
)
def test_classify_reliability(score, expected):
    assert classify_reliability(score) == expected


@pytest.mark.parametrize(
    "invalid_score",
    [
        -0.1,
        1.1,
        True,
    ],
)
def test_classify_invalid_score(invalid_score):
    with pytest.raises(ValueError):
        classify_reliability(invalid_score)


# ==========================================
# SOURCE RELIABILITY CONTRACT TESTS
# ==========================================

def test_source_reliability_contract(valid_factors):
    result = SourceReliability(
        evidence_id="evidence-001",
        reliability_score=0.91,
        reliability_category="HIGH",
        reliability_factors=valid_factors,
        explanation="Recognized source.",
    )

    assert result.evidence_id == "evidence-001"
    assert result.reliability_category == "HIGH"


def test_source_reliability_invalid_category(valid_factors):
    with pytest.raises(ValueError):
        SourceReliability(
            evidence_id="evidence-001",
            reliability_score=0.91,
            reliability_category="TRUSTED",
            reliability_factors=valid_factors,
            explanation="Invalid category.",
        )


def test_source_reliability_missing_factor(valid_factors):
    valid_factors.pop("https")

    with pytest.raises(ValueError):
        SourceReliability(
            evidence_id="evidence-001",
            reliability_score=0.91,
            reliability_category="HIGH",
            reliability_factors=valid_factors,
            explanation="Missing factor.",
        )


# ==========================================
# SINGLE EVIDENCE ANALYSIS
# ==========================================

def test_analyze_nasa_evidence(nasa_evidence):
    result = analyze_source_reliability(nasa_evidence)

    assert isinstance(result, SourceReliability)
    assert result.evidence_id == nasa_evidence.evidence_id
    assert result.reliability_category == "HIGH"
    assert 0.0 <= result.reliability_score <= 1.0


def test_analyze_wikipedia_evidence(wikipedia_evidence):
    result = analyze_source_reliability(wikipedia_evidence)

    assert isinstance(result, SourceReliability)
    assert result.evidence_id == wikipedia_evidence.evidence_id
    assert result.reliability_category == "HIGH"
    assert result.explanation


def test_analyze_unknown_evidence(unknown_evidence):
    result = analyze_source_reliability(unknown_evidence)

    assert isinstance(result, SourceReliability)
    assert result.reliability_category == "MEDIUM"


def test_analyze_invalid_evidence():
    with pytest.raises(ValueError):
        analyze_source_reliability("not evidence")


def test_analyze_invalid_source_url():
    evidence = Evidence(
        claim_id="claim-001",
        passage="Example passage",
        source_title="Example",
        source_url="invalid-url",
        source_type="webpage",
        retrieval_provider="Tavily",
    )

    with pytest.raises(ValueError):
        analyze_source_reliability(evidence)


# ==========================================
# BATCH ANALYSIS
# ==========================================

def test_analyze_sources_batch(
    nasa_evidence,
    wikipedia_evidence,
    unknown_evidence,
):
    evidence_list = [
        nasa_evidence,
        wikipedia_evidence,
        unknown_evidence,
    ]

    results = analyze_sources(evidence_list)

    assert len(results) == 3
    assert all(isinstance(result, SourceReliability) for result in results)

    assert [result.evidence_id for result in results] == [
        evidence.evidence_id for evidence in evidence_list
    ]


def test_analyze_sources_empty_list():
    assert analyze_sources([]) == []


def test_analyze_sources_invalid_input():
    with pytest.raises(ValueError):
        analyze_sources("not a list")


def test_analyze_sources_invalid_item(nasa_evidence):
    with pytest.raises(ValueError):
        analyze_sources([nasa_evidence, "invalid"])