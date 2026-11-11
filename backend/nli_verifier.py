"""
NLI Verification Module

Uses DeBERTa Natural Language Inference to compare
claims against retrieved evidence.

NLI classifications:

1. ENTAILMENT
2. CONTRADICTION
3. NEUTRAL

Important:
NLI predictions represent model confidence,
not guaranteed factual correctness.
"""

import logging
from typing import List

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from backend.data_contracts import (
    Claim,
    EvidenceVerification
)

from backend.evidence_ranker import RankedEvidence


logger = logging.getLogger(__name__)


# ==================================================
# CONFIGURATION
# ==================================================

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

MAX_SEQUENCE_LENGTH = 512

_nli_tokenizer = None
_nli_model = None
_nli_label_mapping = None
_device = None


# ==================================================
# LABEL NORMALIZATION
# ==================================================

def normalize_nli_label(label: str) -> str:
    """
    Normalize model labels into our internal
    NLI classification format.
    """

    if not isinstance(label, str):
        raise ValueError("NLI label must be a string.")

    normalized = label.strip().upper()

    label_mapping = {
        "ENTAILMENT": "ENTAILMENT",
        "CONTRADICTION": "CONTRADICTION",
        "NEUTRAL": "NEUTRAL"
    }

    if normalized not in label_mapping:
        raise ValueError(
            f"Unsupported NLI label: {label}"
        )

    return label_mapping[normalized]


# ==================================================
# MODEL LOADING
# ==================================================

def get_nli_model():
    """
    Load the tokenizer and NLI model lazily.

    The model is loaded only once and reused
    for subsequent predictions.
    """

    global _nli_tokenizer
    global _nli_model
    global _nli_label_mapping
    global _device

    if _nli_model is None or _nli_tokenizer is None:

        logger.info(
            "Loading NLI model: %s",
            NLI_MODEL_NAME
        )

        try:

            _device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                NLI_MODEL_NAME
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL_NAME
            )

            raw_mapping = model.config.id2label

            label_mapping = {}

            for index, label in raw_mapping.items():

                normalized_label = normalize_nli_label(label)

                label_mapping[int(index)] = normalized_label

            expected_labels = {
                "ENTAILMENT",
                "CONTRADICTION",
                "NEUTRAL"
            }

            if set(label_mapping.values()) != expected_labels:
                raise ValueError(
                    "Model must provide exactly the three NLI labels."
                )

            if len(label_mapping) != 3:
                raise ValueError(
                    "NLI model must contain exactly three output classes."
                )

            model.to(_device)

            model.eval()

            _nli_tokenizer = tokenizer
            _nli_model = model
            _nli_label_mapping = label_mapping

            logger.info(
                "NLI model loaded successfully on %s.",
                _device
            )

        except Exception:

            logger.exception("Failed to load NLI model.")

            raise

    return _nli_tokenizer, _nli_model


# ==================================================
# NLI PREDICTION
# ==================================================

def predict_nli(
    premise: str,
    hypothesis: str
) -> dict:
    """
    Compare an evidence passage with a claim.

    Premise:
        Retrieved evidence passage.

    Hypothesis:
        Claim being verified.

    Returns:
        Predicted label and all three probability scores.
    """

    if not isinstance(premise, str) or not premise.strip():
        raise ValueError("Premise cannot be empty.")

    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("Hypothesis cannot be empty.")

    tokenizer, model = get_nli_model()

    inputs = tokenizer(
        premise.strip(),
        hypothesis.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH
    )

    inputs = {
        key: value.to(_device)
        for key, value in inputs.items()
    }

    try:

        with torch.inference_mode():

            outputs = model(**inputs)

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[0]

        scores = {
            _nli_label_mapping[index]: float(probability)
            for index, probability in enumerate(
                probabilities.cpu().tolist()
            )
        }

        predicted_label = max(
            scores,
            key=scores.get
        )

        confidence = scores[predicted_label]

        return {
            "nli_label": predicted_label,
            "nli_confidence": confidence,
            "entailment_score": scores["ENTAILMENT"],
            "contradiction_score": scores["CONTRADICTION"],
            "neutral_score": scores["NEUTRAL"]
        }

    except Exception:

        logger.exception("NLI prediction failed.")

        raise


# ==================================================
# SINGLE EVIDENCE VERIFICATION
# ==================================================

def verify_evidence(
    claim: Claim,
    ranked_evidence: RankedEvidence
) -> EvidenceVerification:
    """
    Verify one ranked evidence passage against a claim.

    Returns an EvidenceVerification object.
    """

    if not isinstance(claim, Claim):
        raise TypeError("claim must be a Claim object.")

    if not isinstance(ranked_evidence, RankedEvidence):
        raise TypeError(
            "ranked_evidence must be a RankedEvidence object."
        )

    evidence = ranked_evidence.evidence

    if evidence.claim_id != claim.claim_id:
        raise ValueError(
            "Evidence claim_id does not match the supplied claim."
        )

    prediction = predict_nli(
        premise=evidence.passage,
        hypothesis=claim.claim_text
    )

    return EvidenceVerification(
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        nli_label=prediction["nli_label"],
        nli_confidence=prediction["nli_confidence"],
        entailment_score=prediction["entailment_score"],
        contradiction_score=prediction["contradiction_score"],
        neutral_score=prediction["neutral_score"]
    )


# ==================================================
# MULTIPLE EVIDENCE VERIFICATION
# ==================================================

def verify_claim(
    claim: Claim,
    ranked_evidence: List[RankedEvidence]
) -> List[EvidenceVerification]:
    """
    Verify all ranked evidence passages associated
    with a single claim.

    Returns:
        A list of EvidenceVerification objects.

    This function does not perform final claim-level
    classification or evidence aggregation.
    """

    if not isinstance(claim, Claim):
        raise TypeError("claim must be a Claim object.")

    if not isinstance(ranked_evidence, list):
        raise TypeError(
            "ranked_evidence must be a list."
        )

    if not ranked_evidence:
        return []

    results = []

    for item in ranked_evidence:

        result = verify_evidence(
            claim,
            item
        )

        results.append(result)

    logger.info(
        "Verified %d evidence passages for claim %s.",
        len(results),
        claim.claim_id
    )

    return results


# ==================================================
# BATCH VERIFICATION
# ==================================================

def verify_claims(
    claims: List[Claim],
    ranked_evidence_by_claim: dict
) -> dict:
    """
    Verify multiple claims.

    Expected input:

    {
        claim_id: [RankedEvidence, RankedEvidence]
    }

    Returns:

    {
        claim_id: [EvidenceVerification, ...]
    }
    """

    if not isinstance(claims, list):
        raise TypeError("claims must be a list.")

    if not isinstance(ranked_evidence_by_claim, dict):
        raise TypeError(
            "ranked_evidence_by_claim must be a dictionary."
        )

    results = {}

    for claim in claims:

        if not isinstance(claim, Claim):
            raise TypeError(
                "Every item in claims must be a Claim object."
            )

        evidence_list = ranked_evidence_by_claim.get(
            claim.claim_id,
            []
        )

        results[claim.claim_id] = verify_claim(
            claim,
            evidence_list
        )

    return results