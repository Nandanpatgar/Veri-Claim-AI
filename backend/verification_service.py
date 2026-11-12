"""
Verification Service

Orchestrates the complete claim-level hallucination detection pipeline.
"""

from backend.input_validator import validate_input_text
from backend.claim_extractor import extract_claims

from backend.evidence_retriever import retrieve_evidence
from backend.evidence_ranker import rank_evidence

from backend.nli_verifier import verify_claim
from backend.reliability_analyzer import analyze_sources

from backend.decision_engine import decide_claim

from backend.report_generator import (
    generate_report,
    calculate_report_metrics,
    serialize_report
)


# ==================================================
# CONFIGURATION
# ==================================================

DEFAULT_TOP_K = 5


# ==================================================
# CUSTOM PIPELINE EXCEPTION
# ==================================================

class VerificationPipelineError(Exception):
    """Raised when a verification pipeline stage fails."""

    def __init__(self, stage, message):
        self.stage = stage
        self.message = message

        super().__init__(
            f"Verification failed during {stage}: {message}"
        )


# ==================================================
# VERIFICATION SERVICE
# ==================================================

class VerificationService:

    def __init__(
        self,
        top_k=DEFAULT_TOP_K
    ):

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        self.top_k = top_k

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------

    def verify(self, text):

        # STEP 1: INPUT VALIDATION

        try:
            original_text = validate_input_text(text)

        except Exception as error:
            raise VerificationPipelineError(
                "input_validation",
                str(error)
            ) from error

        # STEP 2: CLAIM EXTRACTION

        try:
            claims = extract_claims(original_text)

        except Exception as error:
            raise VerificationPipelineError(
                "claim_extraction",
                str(error)
            ) from error

        # STEP 3: HANDLE ZERO CLAIMS

        if not claims:

            report = generate_report(
                original_text,
                [],
                []
            )

            return {
                "report": serialize_report(report),
                "pipeline_status": "completed",
                "retrieval_report": {},
                "message": "No verifiable claims were extracted."
            }

        # STEP 4: EVIDENCE RETRIEVAL

        try:
            retrieval_output = retrieve_evidence(claims)

            evidence_list = retrieval_output["evidence"]

            retrieval_report = retrieval_output[
                "retrieval_report"
            ]

        except Exception as error:
            raise VerificationPipelineError(
                "evidence_retrieval",
                str(error)
            ) from error

        # STEP 5: SOURCE RELIABILITY ANALYSIS

        try:
            reliability_results = analyze_sources(
                evidence_list
            )

        except Exception as error:
            raise VerificationPipelineError(
                "source_reliability",
                str(error)
            ) from error

        # STEP 6: CLAIM-LEVEL RANKING,
        # NLI VERIFICATION AND DECISION

        all_ranked_evidence = []
        all_nli_results = []
        verification_results = []

        for claim in claims:

            # Get evidence belonging to this claim.

            claim_evidence = [
                evidence
                for evidence in evidence_list
                if evidence.claim_id == claim.claim_id
            ]

            # RANK EVIDENCE

            try:
                ranked_evidence = rank_evidence(
                    claim,
                    claim_evidence,
                    top_k=self.top_k
                )

            except Exception as error:
                raise VerificationPipelineError(
                    f"evidence_ranking:{claim.claim_id}",
                    str(error)
                ) from error

            all_ranked_evidence.extend(
                ranked_evidence
            )

            # NLI VERIFICATION

            try:
                nli_results = verify_claim(
                    claim,
                    ranked_evidence
                )

            except Exception as error:
                raise VerificationPipelineError(
                    f"nli_verification:{claim.claim_id}",
                    str(error)
                ) from error

            all_nli_results.extend(nli_results)

            # SELECT RELIABILITY RESULTS
            # Only assessments corresponding to ranked evidence.

            ranked_evidence_ids = {
                item.evidence.evidence_id
                for item in ranked_evidence
            }

            claim_reliability_results = [
                item
                for item in reliability_results
                if item.evidence_id in ranked_evidence_ids
            ]

            # DECISION ENGINE

            try:
                verification_result = decide_claim(
                    claim,
                    ranked_evidence,
                    nli_results,
                    claim_reliability_results
                )

            except Exception as error:
                raise VerificationPipelineError(
                    f"decision_engine:{claim.claim_id}",
                    str(error)
                ) from error

            verification_results.append(
                verification_result
            )

        # STEP 7: REPORT GENERATION

        try:

            report = generate_report(
                original_text,
                claims,
                verification_results
            )

        except Exception as error:
            raise VerificationPipelineError(
                "report_generation",
                str(error)
            ) from error

        # STEP 8: REPORT METRICS

        metrics = calculate_report_metrics(report)

        # STEP 9: REPORT SERIALIZATION

        try:

            report_payload = serialize_report(
                report,
                claims,
                evidence_list,
                all_nli_results,
                reliability_results
            )

        except Exception as error:
            raise VerificationPipelineError(
                "report_serialization",
                str(error)
            ) from error

        # STEP 10: FINAL RESPONSE

        return {
            "report": report_payload,
            "metrics": metrics,
            "retrieval_report": retrieval_report,
            "pipeline_status": "completed"
        }


# ==================================================
# CONVENIENCE FUNCTION
# ==================================================

def verify_text(text, top_k=DEFAULT_TOP_K):

    service = VerificationService(
        top_k=top_k
    )

    return service.verify(text)