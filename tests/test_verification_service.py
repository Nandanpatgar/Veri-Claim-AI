import pytest

from backend import verification_service as service_module
from backend.data_contracts import Claim, VerificationResult


# ==================================================
# TEST FIXTURES
# ==================================================

def create_claim():

    return Claim(
        claim_text="India became independent in 1947.",
        original_text="India became independent in 1947.",
        claim_type="factual",
        claim_id="claim_001"
    )


def setup_successful_pipeline(monkeypatch):

    claim = create_claim()

    verification_result = VerificationResult(
        claim_id=claim.claim_id,
        classification="UNVERIFIABLE",
        nli_label="NEUTRAL",
        nli_confidence=0.95,
        evidence_ids=[],
        explanation="Insufficient evidence."
    )

    fake_report = object()

    # Mock claim extraction
    monkeypatch.setattr(
        service_module,
        "extract_claims",
        lambda text: [claim]
    )

    # Mock evidence retrieval
    monkeypatch.setattr(
        service_module,
        "retrieve_evidence",
        lambda claims: {
            "evidence": [],
            "retrieval_report": {
                "claim_001": {
                    "evidence_count": 0,
                    "errors": {},
                    "provider_status": {}
                }
            }
        }
    )

    # Mock source reliability
    monkeypatch.setattr(
        service_module,
        "analyze_sources",
        lambda evidence: []
    )

    # Mock evidence ranking
    monkeypatch.setattr(
        service_module,
        "rank_evidence",
        lambda claim, evidence, top_k: []
    )

    # Mock NLI verification
    monkeypatch.setattr(
        service_module,
        "verify_claim",
        lambda claim, ranked_evidence: []
    )

    # Mock decision engine
    monkeypatch.setattr(
        service_module,
        "decide_claim",
        lambda claim, ranked, nli, reliability:
            verification_result
    )

    # Mock report generation
    monkeypatch.setattr(
        service_module,
        "generate_report",
        lambda text, claims, results: fake_report
    )

    # Mock report metrics
    monkeypatch.setattr(
        service_module,
        "calculate_report_metrics",
        lambda report: {
            "total_claims": 1,
            "supported_count": 0,
            "contradicted_count": 0,
            "unverifiable_count": 1,
            "hallucination_rate": 0.0,
            "verification_coverage": 0.0
        }
    )

    # Mock report serialization
    monkeypatch.setattr(
        service_module,
        "serialize_report",
        lambda *args: {
            "verification_id": "test_verification",
            "results": []
        }
    )

    return claim


# ==================================================
# 1. INITIALIZATION TESTS
# ==================================================

def test_service_initialization():

    service = service_module.VerificationService()

    assert service.top_k == 5


def test_service_custom_top_k():

    service = service_module.VerificationService(
        top_k=3
    )

    assert service.top_k == 3


def test_service_rejects_invalid_top_k():

    with pytest.raises(ValueError):

        service_module.VerificationService(
            top_k=0
        )


# ==================================================
# 2. INPUT VALIDATION TESTS
# ==================================================

def test_verify_rejects_empty_input():

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("   ")

    assert error.value.stage == "input_validation"


def test_verify_rejects_non_string_input():

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify(123)

    assert error.value.stage == "input_validation"


# ==================================================
# 3. SUCCESSFUL PIPELINE TEST
# ==================================================

def test_successful_verification_pipeline(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    service = service_module.VerificationService()

    result = service.verify(
        "India became independent in 1947."
    )

    assert result["pipeline_status"] == "completed"

    assert "report" in result

    assert "metrics" in result

    assert "retrieval_report" in result


# ==================================================
# 4. EMPTY CLAIM EXTRACTION TEST
# ==================================================

def test_empty_claim_extraction(monkeypatch):

    monkeypatch.setattr(
        service_module,
        "extract_claims",
        lambda text: []
    )

    monkeypatch.setattr(
        service_module,
        "generate_report",
        lambda text, claims, results: object()
    )

    monkeypatch.setattr(
        service_module,
        "serialize_report",
        lambda report: {
            "verification_id": "empty_test",
            "results": []
        }
    )

    service = service_module.VerificationService()

    result = service.verify(
        "Some text without verifiable claims."
    )

    assert result["pipeline_status"] == "completed"

    assert result["message"] == (
        "No verifiable claims were extracted."
    )

    assert result["retrieval_report"] == {}


# ==================================================
# 5. CLAIM EXTRACTION FAILURE
# ==================================================

def test_claim_extraction_failure(monkeypatch):

    def failing_extraction(text):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        service_module,
        "extract_claims",
        failing_extraction
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage == "claim_extraction"


# ==================================================
# 6. EVIDENCE RETRIEVAL FAILURE
# ==================================================

def test_evidence_retrieval_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_retrieval(claims):
        raise RuntimeError("Retrieval unavailable")

    monkeypatch.setattr(
        service_module,
        "retrieve_evidence",
        failing_retrieval
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage == "evidence_retrieval"


# ==================================================
# 7. SOURCE RELIABILITY FAILURE
# ==================================================

def test_source_reliability_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_reliability(evidence):
        raise RuntimeError("Reliability analysis failed")

    monkeypatch.setattr(
        service_module,
        "analyze_sources",
        failing_reliability
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage == "source_reliability"


# ==================================================
# 8. EVIDENCE RANKING FAILURE
# ==================================================

def test_evidence_ranking_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_ranking(claim, evidence, top_k):
        raise RuntimeError("Ranking failed")

    monkeypatch.setattr(
        service_module,
        "rank_evidence",
        failing_ranking
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage.startswith(
        "evidence_ranking:"
    )


# ==================================================
# 9. NLI VERIFICATION FAILURE
# ==================================================

def test_nli_verification_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_nli(claim, ranked_evidence):
        raise RuntimeError("NLI model failed")

    monkeypatch.setattr(
        service_module,
        "verify_claim",
        failing_nli
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage.startswith(
        "nli_verification:"
    )


# ==================================================
# 10. DECISION ENGINE FAILURE
# ==================================================

def test_decision_engine_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_decision(
        claim,
        ranked_evidence,
        nli_results,
        reliability_results
    ):

        raise RuntimeError("Decision failed")

    monkeypatch.setattr(
        service_module,
        "decide_claim",
        failing_decision
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage.startswith(
        "decision_engine:"
    )


# ==================================================
# 11. REPORT GENERATION FAILURE
# ==================================================

def test_report_generation_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_report(text, claims, results):
        raise RuntimeError("Report generation failed")

    monkeypatch.setattr(
        service_module,
        "generate_report",
        failing_report
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage == "report_generation"


# ==================================================
# 12. REPORT SERIALIZATION FAILURE
# ==================================================

def test_report_serialization_failure(monkeypatch):

    setup_successful_pipeline(monkeypatch)

    def failing_serialization(*args):
        raise RuntimeError("Serialization failed")

    monkeypatch.setattr(
        service_module,
        "serialize_report",
        failing_serialization
    )

    service = service_module.VerificationService()

    with pytest.raises(
        service_module.VerificationPipelineError
    ) as error:

        service.verify("Test input")

    assert error.value.stage == "report_serialization"