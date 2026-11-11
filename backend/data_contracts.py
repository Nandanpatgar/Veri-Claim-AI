"""
Data Contracts Module

Defines structured data models used throughout Veri-Claim-AI.

Contracts:
1. Claim
2. Evidence
3. EvidenceVerification
4. VerificationResult
5. VerificationReport
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from uuid import uuid4


# ==========================================
# ALLOWED VALUES
# ==========================================

VALID_CLASSIFICATIONS = {
    "SUPPORTED",
    "CONTRADICTED",
    "UNVERIFIABLE"
}

VALID_NLI_LABELS = {
    "ENTAILMENT",
    "CONTRADICTION",
    "NEUTRAL"
}


# ==========================================
# CLAIM DATA CONTRACT
# ==========================================

@dataclass
class Claim:

    claim_text: str
    original_text: str
    claim_type: str
    claim_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):

        if not isinstance(self.claim_text, str) or not self.claim_text.strip():
            raise ValueError("Claim text cannot be empty.")

        if not isinstance(self.original_text, str) or not self.original_text.strip():
            raise ValueError("Original text cannot be empty.")

        if not isinstance(self.claim_type, str) or not self.claim_type.strip():
            raise ValueError("Claim type cannot be empty.")

        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ValueError("Claim ID cannot be empty.")


# ==========================================
# EVIDENCE DATA CONTRACT
# ==========================================

@dataclass
class Evidence:

    claim_id: str
    passage: str
    source_title: str
    source_url: str
    source_type: str
    retrieval_provider: str
    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    retrieved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):

        required_fields = [
            self.claim_id,
            self.passage,
            self.source_title,
            self.source_url,
            self.source_type,
            self.retrieval_provider
        ]

        if any(
            not isinstance(value, str) or not value.strip()
            for value in required_fields
        ):
            raise ValueError("Evidence fields cannot be empty.")

        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("Evidence ID cannot be empty.")

        if not isinstance(self.retrieved_at, datetime):
            raise ValueError("Retrieved time must be a datetime object.")


# ==========================================
# EVIDENCE VERIFICATION DATA CONTRACT
# ==========================================

@dataclass
class EvidenceVerification:

    claim_id: str
    evidence_id: str

    nli_label: str

    nli_confidence: float

    entailment_score: float
    contradiction_score: float
    neutral_score: float

    def __post_init__(self):

        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ValueError("Claim ID cannot be empty.")

        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("Evidence ID cannot be empty.")

        if self.nli_label not in VALID_NLI_LABELS:
            raise ValueError("Invalid NLI label.")

        scores = [
            self.nli_confidence,
            self.entailment_score,
            self.contradiction_score,
            self.neutral_score
        ]

        for score in scores:

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                raise ValueError(
                    "NLI scores must be finite numbers between 0 and 1."
                )

        total_probability = (
            self.entailment_score
            + self.contradiction_score
            + self.neutral_score
        )

        if abs(total_probability - 1.0) > 0.001:
            raise ValueError("NLI probabilities must sum to 1.")

        expected_confidence = {
            "ENTAILMENT": self.entailment_score,
            "CONTRADICTION": self.contradiction_score,
            "NEUTRAL": self.neutral_score
        }[self.nli_label]

        if abs(self.nli_confidence - expected_confidence) > 0.001:
            raise ValueError(
                "NLI confidence must match the predicted label score."
            )


# ==========================================
# VERIFICATION RESULT CONTRACT
# ==========================================

@dataclass
class VerificationResult:

    claim_id: str
    classification: str
    nli_label: str
    nli_confidence: float
    evidence_ids: list = field(default_factory=list)
    explanation: str = ""

    def __post_init__(self):

        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ValueError("Claim ID cannot be empty.")

        if self.classification not in VALID_CLASSIFICATIONS:
            raise ValueError("Invalid verification classification.")

        if self.nli_label not in VALID_NLI_LABELS:
            raise ValueError("Invalid NLI label.")

        if (
            isinstance(self.nli_confidence, bool)
            or not isinstance(self.nli_confidence, (int, float))
            or not isfinite(self.nli_confidence)
            or not 0.0 <= self.nli_confidence <= 1.0
        ):
            raise ValueError("NLI confidence must be between 0 and 1.")

        if not isinstance(self.evidence_ids, list):
            raise ValueError("Evidence IDs must be a list.")

        if not all(
            isinstance(eid, str) and eid.strip()
            for eid in self.evidence_ids
        ):
            raise ValueError("Every evidence ID must be a non-empty string.")

        if not isinstance(self.explanation, str):
            raise ValueError("Explanation must be a string.")


# ==========================================
# VERIFICATION REPORT CONTRACT
# ==========================================

@dataclass
class VerificationReport:

    original_text: str
    results: list

    verification_id: str = field(default_factory=lambda: str(uuid4()))

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):

        if not isinstance(self.original_text, str):
            raise ValueError("Original text must be a string.")

        if not isinstance(self.results, list):
            raise ValueError("Results must be a list.")

        if not all(
            isinstance(result, VerificationResult)
            for result in self.results
        ):
            raise ValueError(
                "Every report result must be a VerificationResult."
            )

        if not isinstance(self.created_at, datetime):
            raise ValueError("Created time must be a datetime object.")

        claim_ids = [
            result.claim_id
            for result in self.results
        ]

        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Duplicate claim IDs are not allowed.")

    @property
    def total_claims(self):
        return len(self.results)

    @property
    def supported_count(self):
        return sum(
            result.classification == "SUPPORTED"
            for result in self.results
        )

    @property
    def contradicted_count(self):
        return sum(
            result.classification == "CONTRADICTED"
            for result in self.results
        )

    @property
    def unverifiable_count(self):
        return sum(
            result.classification == "UNVERIFIABLE"
            for result in self.results
        )