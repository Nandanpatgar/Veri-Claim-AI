import json

import pytest

from backend.data_contracts import (
    Claim,
    Evidence,
    EvidenceVerification,
    SourceReliability,
    VerificationResult,
)

from backend.report_generator import (
    generate_report,
    calculate_report_metrics,
    build_report_details,
    serialize_report,
    export_report_json,
)


# ==================================================
# TEST DATA
# ==================================================

def create_claim(text, claim_type="factual"):

    return Claim(
        claim_text=text,
        original_text=text,
        claim_type=claim_type,
    )


def create_result(claim, classification):

    labels = {
        "SUPPORTED": "ENTAILMENT",
        "CONTRADICTED": "CONTRADICTION",
        "UNVERIFIABLE": "NEUTRAL",
    }

    return VerificationResult(
        claim_id=claim.claim_id,
        classification=classification,
        nli_label=labels[classification],
        nli_confidence=0.95,
        evidence_ids=[],
        explanation=f"Test explanation for {classification}.",
    )


def create_evidence(claim):

    return Evidence(
        claim_id=claim.claim_id,
        passage="The Earth revolves around the Sun.",
        source_title="Scientific Source",
        source_url="https://example.com/article",
        source_type="research",
        retrieval_provider="Test",
    )


def create_nli_result(claim, evidence):

    return EvidenceVerification(
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        nli_label="ENTAILMENT",
        nli_confidence=0.90,
        entailment_score=0.90,
        contradiction_score=0.05,
        neutral_score=0.05,
    )


def create_reliability_result(evidence):

    return SourceReliability(
        evidence_id=evidence.evidence_id,
        reliability_score=0.90,
        reliability_category="HIGH",
        reliability_factors={
            "domain_reputation": 0.90,
            "source_category": 0.90,
            "https": 1.0,
            "publisher_identification": 0.90,
            "url_transparency": 0.90,
        },
        explanation="Reliable test source.",
    )


# ==================================================
# REPORT GENERATION TESTS
# ==================================================

def test_generate_report_success():

    claim = create_claim("Earth revolves around the Sun.")

    result = create_result(claim, "SUPPORTED")

    report = generate_report(
        "Earth revolves around the Sun.",
        [claim],
        [result],
    )

    assert report.total_claims == 1
    assert report.supported_count == 1
    assert report.contradicted_count == 0
    assert report.unverifiable_count == 0


def test_generate_report_multiple_claims():

    claims = [
        create_claim("Claim one"),
        create_claim("Claim two"),
        create_claim("Claim three"),
    ]

    results = [
        create_result(claims[0], "SUPPORTED"),
        create_result(claims[1], "CONTRADICTED"),
        create_result(claims[2], "UNVERIFIABLE"),
    ]

    report = generate_report(
        "Original input",
        claims,
        results,
    )

    assert report.total_claims == 3
    assert report.supported_count == 1
    assert report.contradicted_count == 1
    assert report.unverifiable_count == 1


def test_generate_empty_report():

    report = generate_report(
        "",
        [],
        [],
    )

    assert report.total_claims == 0


def test_generate_report_invalid_original_text():

    with pytest.raises(TypeError):

        generate_report(
            123,
            [],
            [],
        )


def test_generate_report_invalid_claims():

    with pytest.raises(TypeError):

        generate_report(
            "Input",
            ["invalid"],
            [],
        )


def test_generate_report_invalid_results():

    with pytest.raises(TypeError):

        generate_report(
            "Input",
            [],
            ["invalid"],
        )


def test_generate_report_missing_verification_result():

    claim = create_claim("Test claim")

    with pytest.raises(ValueError):

        generate_report(
            "Input",
            [claim],
            [],
        )


def test_generate_report_duplicate_claim_ids():

    claim = create_claim("Test claim")

    with pytest.raises(ValueError):

        generate_report(
            "Input",
            [claim, claim],
            [
                create_result(claim, "SUPPORTED"),
            ],
        )


# ==================================================
# METRICS TESTS
# ==================================================

def test_calculate_metrics():

    claims = [
        create_claim("Claim one"),
        create_claim("Claim two"),
        create_claim("Claim three"),
        create_claim("Claim four"),
    ]

    results = [
        create_result(claims[0], "SUPPORTED"),
        create_result(claims[1], "SUPPORTED"),
        create_result(claims[2], "CONTRADICTED"),
        create_result(claims[3], "UNVERIFIABLE"),
    ]

    report = generate_report(
        "Original input",
        claims,
        results,
    )

    metrics = calculate_report_metrics(report)

    assert metrics["total_claims"] == 4
    assert metrics["supported_count"] == 2
    assert metrics["contradicted_count"] == 1
    assert metrics["unverifiable_count"] == 1

    assert metrics["hallucination_rate"] == 25.0
    assert metrics["verification_coverage"] == 75.0


def test_metrics_for_empty_report():

    report = generate_report("", [], [])

    metrics = calculate_report_metrics(report)

    assert metrics["hallucination_rate"] == 0.0
    assert metrics["verification_coverage"] == 0.0


def test_metrics_invalid_report():

    with pytest.raises(TypeError):

        calculate_report_metrics("invalid")


# ==================================================
# CLAIM DETAIL TESTS
# ==================================================

def test_build_report_details():

    claim = create_claim("Earth revolves around the Sun.")

    evidence = create_evidence(claim)

    nli = create_nli_result(claim, evidence)

    reliability = create_reliability_result(evidence)

    result = VerificationResult(
        claim_id=claim.claim_id,
        classification="SUPPORTED",
        nli_label="ENTAILMENT",
        nli_confidence=0.95,
        evidence_ids=[evidence.evidence_id],
        explanation="Evidence supports the claim.",
    )

    report = generate_report(
        claim.original_text,
        [claim],
        [result],
    )

    details = build_report_details(
        report,
        [claim],
        [evidence],
        [nli],
        [reliability],
    )

    assert len(details) == 1

    assert details[0]["classification"] == "SUPPORTED"

    assert details[0]["evidence_count"] == 1

    assert details[0]["evidence"][0]["nli_assessment"]["label"] == "ENTAILMENT"

    assert (
        details[0]["evidence"][0]["source_reliability"]["category"]
        == "HIGH"
    )


def test_build_details_invalid_report():

    with pytest.raises(TypeError):

        build_report_details(
            "invalid",
            [],
            [],
            [],
            [],
        )


def test_build_details_unknown_evidence():

    claim = create_claim("Test claim")

    result = VerificationResult(
        claim_id=claim.claim_id,
        classification="SUPPORTED",
        nli_label="ENTAILMENT",
        nli_confidence=0.90,
        evidence_ids=["unknown-evidence"],
        explanation="Test",
    )

    report = generate_report(
        "Input",
        [claim],
        [result],
    )

    with pytest.raises(ValueError):

        build_report_details(
            report,
            [claim],
            [],
            [],
            [],
        )


# ==================================================
# SERIALIZATION TESTS
# ==================================================

def test_serialize_report():

    claim = create_claim("Test claim")

    result = create_result(claim, "SUPPORTED")

    report = generate_report(
        "Original input",
        [claim],
        [result],
    )

    payload = serialize_report(report)

    assert "verification_id" in payload
    assert "created_at" in payload
    assert "original_text" in payload
    assert "summary" in payload
    assert "results" in payload


def test_serialized_report_is_json_compatible():

    claim = create_claim("Test claim")

    result = create_result(claim, "SUPPORTED")

    report = generate_report(
        "Original input",
        [claim],
        [result],
    )

    payload = serialize_report(report)

    json_string = json.dumps(payload)

    assert isinstance(json_string, str)


def test_serialize_report_with_details():

    claim = create_claim("Test claim")

    evidence = create_evidence(claim)

    nli = create_nli_result(claim, evidence)

    reliability = create_reliability_result(evidence)

    result = VerificationResult(
        claim_id=claim.claim_id,
        classification="SUPPORTED",
        nli_label="ENTAILMENT",
        nli_confidence=0.95,
        evidence_ids=[evidence.evidence_id],
        explanation="Supported by evidence.",
    )

    report = generate_report(
        "Test input",
        [claim],
        [result],
    )

    payload = serialize_report(
        report,
        [claim],
        [evidence],
        [nli],
        [reliability],
    )

    assert "claim_details" in payload
    assert len(payload["claim_details"]) == 1


def test_serialize_rejects_partial_details():

    report = generate_report("", [], [])

    with pytest.raises(ValueError):

        serialize_report(
            report,
            claims=[],
        )


# ==================================================
# JSON EXPORT TESTS
# ==================================================

def test_export_report_json(tmp_path):

    claim = create_claim("Test claim")

    result = create_result(claim, "SUPPORTED")

    report = generate_report(
        "Original input",
        [claim],
        [result],
    )

    output_file = tmp_path / "report.json"

    returned_path = export_report_json(
        report,
        output_file,
    )

    assert returned_path.exists()

    with output_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["summary"]["total_claims"] == 1


def test_export_report_creates_directory(tmp_path):

    report = generate_report("", [], [])

    output_file = tmp_path / "reports" / "nested" / "report.json"

    export_report_json(
        report,
        output_file,
    )

    assert output_file.exists()


def test_export_invalid_path_type():

    report = generate_report("", [], [])

    with pytest.raises(TypeError):

        export_report_json(
            report,
            123,
        )