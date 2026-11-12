import pytest

from backend.data_contracts import (
    Claim,
    Evidence,
    EvidenceVerification,
    SourceReliability,
)

from backend.evidence_ranker import RankedEvidence

from backend.decision_engine import (
    decide_claim,
    get_source_domain,
)


# --------------------------------------------------
# TEST DATA FACTORIES
# --------------------------------------------------

def create_claim():
    return Claim(
        claim_text="The Earth revolves around the Sun.",
        original_text="The Earth revolves around the Sun.",
        claim_type="factual",
    )


def create_evidence(claim, domain, index):

    return Evidence(
        claim_id=claim.claim_id,
        passage=f"Evidence passage {index}",
        source_title=f"Source {index}",
        source_url=f"https://www.{domain}/article-{index}",
        source_type="research",
        retrieval_provider="Test",
    )


def create_ranked_evidence(evidence, score=0.80):

    return RankedEvidence(
        evidence=evidence,
        rank=1,
        bm25_score=0.80,
        semantic_score=0.80,
        combined_score=score,
    )


def create_nli_result(
    claim,
    evidence,
    label="ENTAILMENT",
    confidence=0.95,
):

    if label == "ENTAILMENT":
        entailment = confidence
        contradiction = (1 - confidence) / 2
        neutral = (1 - confidence) / 2

    elif label == "CONTRADICTION":
        contradiction = confidence
        entailment = (1 - confidence) / 2
        neutral = (1 - confidence) / 2

    else:
        neutral = confidence
        entailment = (1 - confidence) / 2
        contradiction = (1 - confidence) / 2

    return EvidenceVerification(
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        nli_label=label,
        nli_confidence=confidence,
        entailment_score=entailment,
        contradiction_score=contradiction,
        neutral_score=neutral,
    )


def create_reliability_result(evidence, score=0.90):

    return SourceReliability(
        evidence_id=evidence.evidence_id,
        reliability_score=score,
        reliability_category="HIGH" if score >= 0.80 else "MEDIUM",
        reliability_factors={
            "domain_reputation": score,
            "source_category": score,
            "https": 1.0,
            "publisher_identification": score,
            "url_transparency": score,
        },
        explanation="Test reliability assessment.",
    )


def build_inputs(
    claim,
    domains,
    labels=None,
    relevance_scores=None,
    reliability_scores=None,
    nli_confidences=None,
):

    labels = labels or ["ENTAILMENT"] * len(domains)
    relevance_scores = relevance_scores or [0.80] * len(domains)
    reliability_scores = reliability_scores or [0.90] * len(domains)
    nli_confidences = nli_confidences or [0.95] * len(domains)

    ranked = []
    nli_results = []
    reliability_results = []

    for index, domain in enumerate(domains):

        evidence = create_evidence(claim, domain, index)

        ranked.append(
            create_ranked_evidence(
                evidence,
                relevance_scores[index],
            )
        )

        nli_results.append(
            create_nli_result(
                claim,
                evidence,
                labels[index],
                nli_confidences[index],
            )
        )

        reliability_results.append(
            create_reliability_result(
                evidence,
                reliability_scores[index],
            )
        )

    return ranked, nli_results, reliability_results


# --------------------------------------------------
# SOURCE DOMAIN TESTS
# --------------------------------------------------

def test_normalizes_www_domain():

    assert (
        get_source_domain("https://www.example.com/article")
        == "example.com"
    )


def test_normalizes_subdomains():

    assert (
        get_source_domain("https://news.example.com/article")
        == "example.com"
    )


def test_handles_multi_label_domains():

    assert (
        get_source_domain("https://news.example.co.uk/article")
        == "example.co.uk"
    )


# --------------------------------------------------
# CLASSIFICATION TESTS
# --------------------------------------------------

def test_classifies_supported_claim():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "SUPPORTED"
    assert result.nli_label == "ENTAILMENT"
    assert len(result.evidence_ids) == 2


def test_classifies_contradicted_claim():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
        labels=["CONTRADICTION", "CONTRADICTION"],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "CONTRADICTED"
    assert result.nli_label == "CONTRADICTION"


def test_classifies_single_source_as_unverifiable():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com"],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"


def test_classifies_no_evidence_as_unverifiable():

    claim = create_claim()

    result = decide_claim(
        claim,
        [],
        [],
        [],
    )

    assert result.classification == "UNVERIFIABLE"
    assert result.evidence_ids == []


# --------------------------------------------------
# EVIDENCE QUALITY TESTS
# --------------------------------------------------

def test_low_relevance_evidence_is_excluded():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
        relevance_scores=[0.30, 0.30],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"
    assert result.evidence_ids == []


def test_low_reliability_evidence_is_excluded():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
        reliability_scores=[0.40, 0.40],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"
    assert result.evidence_ids == []


def test_low_nli_confidence_is_not_decisive():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
        nli_confidences=[0.70, 0.75],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"


def test_neutral_predictions_are_not_decisive():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
        labels=["NEUTRAL", "NEUTRAL"],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"


# --------------------------------------------------
# INDEPENDENCE AND CONFLICT TESTS
# --------------------------------------------------

def test_subdomains_are_not_independent_sources():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        [
            "news.example.com",
            "research.example.com",
        ],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"


def test_supporting_and_contradictory_sources_create_conflict():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        [
            "source-one.com",
            "source-two.com",
            "source-three.com",
        ],
        labels=[
            "ENTAILMENT",
            "ENTAILMENT",
            "CONTRADICTION",
        ],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"
    assert "conflicting" in result.explanation.lower()


def test_same_domain_conflict_is_not_counted():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        [
            "news.example.com",
            "research.example.com",
            "source-two.com",
        ],
        labels=[
            "ENTAILMENT",
            "CONTRADICTION",
            "ENTAILMENT",
        ],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert result.classification == "UNVERIFIABLE"


# --------------------------------------------------
# MISSING ASSESSMENT TESTS
# --------------------------------------------------

def test_missing_nli_assessment_is_excluded():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
    )

    result = decide_claim(
        claim,
        ranked,
        nli[:1],
        reliability,
    )

    assert result.classification == "UNVERIFIABLE"
    assert len(result.evidence_ids) == 1


def test_missing_reliability_assessment_is_excluded():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
    )

    result = decide_claim(
        claim,
        ranked,
        nli,
        reliability[:1],
    )

    assert result.classification == "UNVERIFIABLE"
    assert len(result.evidence_ids) == 1


# --------------------------------------------------
# VALIDATION TESTS
# --------------------------------------------------

def test_rejects_invalid_claim():

    with pytest.raises(TypeError):

        decide_claim(
            "This is not a Claim object",
            [],
            [],
            [],
        )


def test_rejects_invalid_ranked_evidence_type():

    claim = create_claim()

    with pytest.raises(TypeError):

        decide_claim(
            claim,
            ["invalid"],
            [],
            [],
        )


def test_rejects_nli_result_for_another_claim():

    claim = create_claim()
    another_claim = create_claim()

    ranked, _, reliability = build_inputs(
        claim,
        ["source-one.com"],
    )

    incorrect_nli = create_nli_result(
        another_claim,
        ranked[0].evidence,
    )

    with pytest.raises(ValueError):

        decide_claim(
            claim,
            ranked,
            [incorrect_nli],
            reliability,
        )


def test_rejects_duplicate_ranked_evidence():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com"],
    )

    with pytest.raises(ValueError):

        decide_claim(
            claim,
            [ranked[0], ranked[0]],
            nli,
            reliability,
        )


def test_rejects_duplicate_nli_assessment():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com"],
    )

    with pytest.raises(ValueError):

        decide_claim(
            claim,
            ranked,
            [nli[0], nli[0]],
            reliability,
        )


def test_explanation_is_generated():

    claim = create_claim()

    ranked, nli, reliability = build_inputs(
        claim,
        ["source-one.com", "source-two.com"],
    )

    result = decide_claim(claim, ranked, nli, reliability)

    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0