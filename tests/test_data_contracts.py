import pytest

from backend.data_contracts import (
    Claim,
    Evidence,
    VerificationResult,
    VerificationReport
)


# ==========================================
# CLAIM TESTS
# ==========================================

def test_valid_claim():

    claim = Claim(
        claim_text="India gained independence in 1947.",
        original_text="India gained independence in 1947.",
        claim_type="historical"
    )

    assert claim.claim_text == "India gained independence in 1947."
    assert claim.claim_id is not None


def test_empty_claim_text():

    with pytest.raises(ValueError):
        Claim("", "Original text", "historical")


def test_empty_original_text():

    with pytest.raises(ValueError):
        Claim("Claim text", "", "historical")


# ==========================================
# EVIDENCE TESTS
# ==========================================

def test_valid_evidence():

    evidence = Evidence(
        claim_id="claim-001",
        passage="India became independent in 1947.",
        source_title="History Reference",
        source_url="https://example.com",
        source_type="encyclopedia",
        retrieval_provider="Wikipedia"
    )

    assert evidence.claim_id == "claim-001"
    assert evidence.evidence_id is not None


def test_empty_evidence_passage():

    with pytest.raises(ValueError):
        Evidence(
            claim_id="claim-001",
            passage="",
            source_title="History Reference",
            source_url="https://example.com",
            source_type="encyclopedia",
            retrieval_provider="Wikipedia"
        )


# ==========================================
# VERIFICATION RESULT TESTS
# ==========================================

def test_valid_verification_result():

    result = VerificationResult(
        claim_id="claim-001",
        classification="SUPPORTED",
        nli_label="ENTAILMENT",
        nli_confidence=0.95,
        evidence_ids=["evidence-001"],
        explanation="Evidence supports the claim."
    )

    assert result.classification == "SUPPORTED"
    assert result.nli_confidence == 0.95


def test_invalid_classification():

    with pytest.raises(ValueError):
        VerificationResult(
            claim_id="claim-001",
            classification="UNKNOWN",
            nli_label="ENTAILMENT",
            nli_confidence=0.95
        )


def test_invalid_nli_label():

    with pytest.raises(ValueError):
        VerificationResult(
            claim_id="claim-001",
            classification="SUPPORTED",
            nli_label="UNKNOWN",
            nli_confidence=0.95
        )


def test_confidence_out_of_range():

    with pytest.raises(ValueError):
        VerificationResult(
            claim_id="claim-001",
            classification="SUPPORTED",
            nli_label="ENTAILMENT",
            nli_confidence=1.5
        )


def test_confidence_boundary_values():

    for confidence in [0.0, 1.0]:

        result = VerificationResult(
            claim_id="claim-001",
            classification="SUPPORTED",
            nli_label="ENTAILMENT",
            nli_confidence=confidence
        )

        assert result.nli_confidence == confidence


# ==========================================
# VERIFICATION REPORT TESTS
# ==========================================

def test_valid_verification_report():

    result = VerificationResult(
        claim_id="claim-001",
        classification="SUPPORTED",
        nli_label="ENTAILMENT",
        nli_confidence=0.95
    )

    report = VerificationReport(
        original_text="India gained independence in 1947.",
        results=[result]
    )

    assert report.total_claims == 1
    assert report.supported_count == 1


def test_report_classification_counts():

    results = [
        VerificationResult(
            claim_id="claim-001",
            classification="SUPPORTED",
            nli_label="ENTAILMENT",
            nli_confidence=0.95
        ),
        VerificationResult(
            claim_id="claim-002",
            classification="CONTRADICTED",
            nli_label="CONTRADICTION",
            nli_confidence=0.90
        ),
        VerificationResult(
            claim_id="claim-003",
            classification="UNVERIFIABLE",
            nli_label="NEUTRAL",
            nli_confidence=0.60
        )
    ]

    report = VerificationReport(
        original_text="Sample text",
        results=results
    )

    assert report.total_claims == 3
    assert report.supported_count == 1
    assert report.contradicted_count == 1
    assert report.unverifiable_count == 1


def test_duplicate_claim_ids():

    result1 = VerificationResult(
        claim_id="claim-001",
        classification="SUPPORTED",
        nli_label="ENTAILMENT",
        nli_confidence=0.95
    )

    result2 = VerificationResult(
        claim_id="claim-001",
        classification="CONTRADICTED",
        nli_label="CONTRADICTION",
        nli_confidence=0.90
    )

    with pytest.raises(ValueError):
        VerificationReport(
            original_text="Sample text",
            results=[result1, result2]
        )


def test_empty_report():

    report = VerificationReport(
        original_text="No factual claims found.",
        results=[]
    )

    assert report.total_claims == 0
    assert report.supported_count == 0
    assert report.contradicted_count == 0
    assert report.unverifiable_count == 0