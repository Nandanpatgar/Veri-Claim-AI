"""
Report Generator
----------------

Generates structured, explainable verification reports
from claim-level verification results.

Responsibilities:
    1. Create VerificationReport objects.
    2. Calculate verification metrics.
    3. Attach claim-level evidence details.
    4. Include NLI and source reliability assessments.
    5. Serialize reports into JSON.
    6. Export reports to files.
"""

import json

from pathlib import Path
from datetime import datetime, timezone

from backend.data_contracts import (
    Claim,
    Evidence,
    EvidenceVerification,
    SourceReliability,
    VerificationResult,
    VerificationReport,
)


# ==================================================
# REPORT GENERATION
# ==================================================

def generate_report(original_text, claims, verification_results):
    """
    Create a VerificationReport.

    Args:
        original_text: Original user-submitted text.
        claims: List of extracted Claim objects.
        verification_results: List of VerificationResult objects.

    Returns:
        VerificationReport
    """

    if not isinstance(original_text, str):
        raise TypeError("Original text must be a string.")

    if not isinstance(claims, list):
        raise TypeError("Claims must be a list.")

    if not isinstance(verification_results, list):
        raise TypeError("Verification results must be a list.")

    if not all(isinstance(claim, Claim) for claim in claims):
        raise TypeError("Every claim must be a Claim object.")

    if not all(
        isinstance(result, VerificationResult)
        for result in verification_results
    ):
        raise TypeError(
            "Every result must be a VerificationResult object."
        )

    claim_ids = [claim.claim_id for claim in claims]

    result_ids = [
        result.claim_id
        for result in verification_results
    ]

    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Duplicate claim IDs are not allowed.")

    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Duplicate result IDs are not allowed.")

    if set(claim_ids) != set(result_ids):
        raise ValueError(
            "Every extracted claim must have exactly one verification result."
        )

    return VerificationReport(
        original_text=original_text,
        results=verification_results,
    )


# ==================================================
# REPORT METRICS
# ==================================================

def calculate_report_metrics(report):
    """
    Calculate classification counts and verification metrics.

    Hallucination rate:
        Contradicted claims / Total claims * 100

    Verification coverage:
        (Supported + Contradicted) / Total claims * 100

    Unverifiable claims are not automatically considered
    hallucinations.
    """

    if not isinstance(report, VerificationReport):
        raise TypeError("report must be a VerificationReport object.")

    total = report.total_claims

    supported = report.supported_count
    contradicted = report.contradicted_count
    unverifiable = report.unverifiable_count

    if total == 0:
        hallucination_rate = 0.0
        verification_coverage = 0.0
    else:
        hallucination_rate = (
            contradicted / total
        ) * 100

        verification_coverage = (
            (supported + contradicted) / total
        ) * 100

    return {
        "total_claims": total,
        "supported_count": supported,
        "contradicted_count": contradicted,
        "unverifiable_count": unverifiable,
        "hallucination_rate": round(hallucination_rate, 2),
        "verification_coverage": round(
            verification_coverage, 2
        ),
    }


# ==================================================
# CLAIM-WISE REPORT DETAILS
# ==================================================

def build_report_details(
    report,
    claims,
    evidence_list,
    nli_results,
    reliability_results,
):
    """
    Build detailed claim-wise report information.

    Combines:
        Claim
        VerificationResult
        Evidence
        EvidenceVerification
        SourceReliability
    """

    if not isinstance(report, VerificationReport):
        raise TypeError("Invalid verification report.")

    if not isinstance(claims, list):
        raise TypeError("Claims must be a list.")

    if not isinstance(evidence_list, list):
        raise TypeError("Evidence list must be a list.")

    if not isinstance(nli_results, list):
        raise TypeError("NLI results must be a list.")

    if not isinstance(reliability_results, list):
        raise TypeError("Reliability results must be a list.")

    if not all(isinstance(item, Claim) for item in claims):
        raise TypeError("Invalid Claim object.")

    if not all(
        isinstance(item, Evidence)
        for item in evidence_list
    ):
        raise TypeError("Invalid Evidence object.")

    if not all(
        isinstance(item, EvidenceVerification)
        for item in nli_results
    ):
        raise TypeError("Invalid NLI result.")

    if not all(
        isinstance(item, SourceReliability)
        for item in reliability_results
    ):
        raise TypeError("Invalid reliability result.")

    claim_map = {
        claim.claim_id: claim
        for claim in claims
    }

    evidence_map = {
        evidence.evidence_id: evidence
        for evidence in evidence_list
    }

    nli_map = {
        result.evidence_id: result
        for result in nli_results
    }

    reliability_map = {
        result.evidence_id: result
        for result in reliability_results
    }

    if len(claim_map) != len(claims):
        raise ValueError("Duplicate claim IDs.")

    if len(evidence_map) != len(evidence_list):
        raise ValueError("Duplicate evidence IDs.")

    if len(nli_map) != len(nli_results):
        raise ValueError("Duplicate NLI assessment IDs.")

    if len(reliability_map) != len(reliability_results):
        raise ValueError("Duplicate reliability assessment IDs.")

    for evidence in evidence_list:
        if evidence.claim_id not in claim_map:
            raise ValueError(
                "Evidence references an unknown claim."
            )

    for result in nli_results:
        if result.evidence_id not in evidence_map:
            raise ValueError(
                "NLI result references unknown evidence."
            )

        if (
            result.claim_id
            != evidence_map[result.evidence_id].claim_id
        ):
            raise ValueError(
                "NLI result and evidence claim IDs do not match."
            )

    for result in reliability_results:
        if result.evidence_id not in evidence_map:
            raise ValueError(
                "Reliability result references unknown evidence."
            )

    claim_details = []

    for verification in report.results:

        claim = claim_map.get(verification.claim_id)

        if claim is None:
            raise ValueError(
                "Verification result references an unknown claim."
            )

        claim_evidence = []

        for evidence_id in verification.evidence_ids:

            evidence = evidence_map.get(evidence_id)

            if evidence is None:
                raise ValueError(
                    "Verification result references unknown evidence."
                )

            if evidence.claim_id != claim.claim_id:
                raise ValueError(
                    "Evidence belongs to a different claim."
                )

            nli = nli_map.get(evidence_id)
            reliability = reliability_map.get(evidence_id)

            claim_evidence.append({
                "evidence_id": evidence.evidence_id,
                "passage": evidence.passage,
                "source_title": evidence.source_title,
                "source_url": evidence.source_url,
                "source_type": evidence.source_type,
                "retrieval_provider": evidence.retrieval_provider,
                "retrieved_at": evidence.retrieved_at.isoformat(),

                "nli_assessment": (
                    {
                        "label": nli.nli_label,
                        "confidence": nli.nli_confidence,
                        "entailment_score": nli.entailment_score,
                        "contradiction_score": nli.contradiction_score,
                        "neutral_score": nli.neutral_score,
                    }
                    if nli else None
                ),

                "source_reliability": (
                    {
                        "score": reliability.reliability_score,
                        "category": reliability.reliability_category,
                        "factors": reliability.reliability_factors,
                        "explanation": reliability.explanation,
                    }
                    if reliability else None
                ),
            })

        claim_details.append({
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "claim_type": claim.claim_type,

            "classification": verification.classification,

            "nli_label": verification.nli_label,
            "nli_confidence": verification.nli_confidence,

            "explanation": verification.explanation,

            "evidence_count": len(claim_evidence),
            "evidence": claim_evidence,
        })

    return claim_details


# ==================================================
# REPORT SERIALIZATION
# ==================================================

def serialize_report(
    report,
    claims=None,
    evidence_list=None,
    nli_results=None,
    reliability_results=None,
):
    """
    Convert a VerificationReport into a JSON-compatible dictionary.

    Detailed claim information is included when all supporting
    data collections are supplied.
    """

    if not isinstance(report, VerificationReport):
        raise TypeError("Invalid VerificationReport object.")

    payload = {
        "verification_id": report.verification_id,

        "created_at": report.created_at.isoformat(),

        "original_text": report.original_text,

        "summary": calculate_report_metrics(report),

        "results": [
            {
                "claim_id": result.claim_id,
                "classification": result.classification,
                "nli_label": result.nli_label,
                "nli_confidence": result.nli_confidence,
                "evidence_ids": result.evidence_ids,
                "explanation": result.explanation,
            }
            for result in report.results
        ],
    }

    optional_data = [
        claims,
        evidence_list,
        nli_results,
        reliability_results,
    ]

    if any(item is not None for item in optional_data):

        if any(item is None for item in optional_data):
            raise ValueError(
                "All four detailed report inputs must be provided together."
            )

        payload["claim_details"] = build_report_details(
            report,
            claims,
            evidence_list,
            nli_results,
            reliability_results,
        )

    return payload


# ==================================================
# JSON EXPORT
# ==================================================

def export_report_json(
    report,
    output_path,
    claims=None,
    evidence_list=None,
    nli_results=None,
    reliability_results=None,
):
    """
    Export a verification report as a JSON file.

    Returns:
        Path of the generated JSON file.
    """

    if not isinstance(output_path, (str, Path)):
        raise TypeError("Output path must be a string or Path.")

    payload = serialize_report(
        report,
        claims,
        evidence_list,
        nli_results,
        reliability_results,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:

        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path