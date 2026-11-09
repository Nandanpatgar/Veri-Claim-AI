
"""
Evidence Ranking Module

Ranks retrieved evidence using:
1. BM25 keyword relevance
2. Sentence Transformer semantic similarity
3. Weighted combined relevance score

Provides explainable ranking results containing
individual scores and ranking positions.

Important:
Evidence ranking measures relevance, NOT factual correctness.
Final claim verification is handled by the NLI module.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from backend.data_contracts import Claim, Evidence


logger = logging.getLogger(__name__)


# ==================================================
# CONFIGURATION
# ==================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

DEFAULT_TOP_K = 5

_embedding_model = None


# ==================================================
# RANKED EVIDENCE DATA CONTRACT
# ==================================================

@dataclass
class RankedEvidence:
    """
    Stores evidence along with its ranking information.

    Scores are relevance measurements, not truth
    or factual confidence measurements.
    """

    evidence: Evidence
    rank: int
    bm25_score: float
    semantic_score: float
    combined_score: float


# ==================================================
# TEXT PREPROCESSING
# ==================================================

def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase tokens.
    """

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    return re.findall(r"\b\w+\b", text.lower())


# ==================================================
# EMBEDDING MODEL
# ==================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Load the Sentence Transformer model lazily.

    Reuses the model after its first initialization.
    """

    global _embedding_model

    if _embedding_model is None:

        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL_NAME
        )

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


# ==================================================
# BM25 SCORING
# ==================================================

def calculate_bm25_scores(
    claim_text: str,
    evidence_list: List[Evidence]
) -> List[float]:
    """
    Calculate BM25 keyword relevance scores.
    """

    if not evidence_list:
        return []

    tokenized_documents = [
        tokenize(evidence.passage)
        for evidence in evidence_list
    ]

    tokenized_claim = tokenize(claim_text)

    if not tokenized_claim:
        return [0.0] * len(evidence_list)

    bm25 = BM25Okapi(tokenized_documents)

    scores = bm25.get_scores(tokenized_claim)

    return [
        float(score)
        for score in scores
    ]


# ==================================================
# SCORE NORMALIZATION
# ==================================================

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to the range [0, 1].

    If all scores are identical, return zeros.
    """

    if not scores:
        return []

    scores_array = np.asarray(scores, dtype=float)

    if not np.all(np.isfinite(scores_array)):
        raise ValueError("Scores must contain finite values.")

    minimum = float(np.min(scores_array))
    maximum = float(np.max(scores_array))

    if maximum == minimum:
        return [0.0] * len(scores)

    normalized = (
        (scores_array - minimum) /
        (maximum - minimum)
    )

    return normalized.tolist()


# ==================================================
# SEMANTIC SIMILARITY
# ==================================================

def calculate_semantic_scores(
    claim_text: str,
    evidence_list: List[Evidence]
) -> List[float]:
    """
    Calculate cosine similarity between the claim
    and each evidence passage.
    """

    if not evidence_list:
        return []

    model = get_embedding_model()

    passages = [
        evidence.passage
        for evidence in evidence_list
    ]

    claim_embedding = model.encode(
        claim_text,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    evidence_embeddings = model.encode(
        passages,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        claim_embedding,
        evidence_embeddings
    )[0]

    scores = [
        float(score)
        for score in similarities.cpu().tolist()
    ]

    if not all(np.isfinite(score) for score in scores):
        raise ValueError("Semantic scores must be finite.")

    return scores


# ==================================================
# COMBINED SCORING
# ==================================================

def calculate_combined_scores(
    bm25_scores: List[float],
    semantic_scores: List[float]
) -> List[float]:
    """
    Combine normalized BM25 and semantic scores.

    Formula:

    R = 0.4 * BM25_normalized
        + 0.6 * Semantic_similarity
    """

    if len(bm25_scores) != len(semantic_scores):
        raise ValueError(
            "BM25 and semantic score counts must match."
        )

    if not all(
        np.isfinite(score)
        for score in semantic_scores
    ):
        raise ValueError("Semantic scores must be finite.")

    normalized_bm25 = normalize_scores(bm25_scores)

    combined_scores = []

    for bm25_score, semantic_score in zip(
        normalized_bm25,
        semantic_scores
    ):

        combined = (
            BM25_WEIGHT * bm25_score +
            SEMANTIC_WEIGHT * semantic_score
        )

        combined_scores.append(float(combined))

    return combined_scores


# ==================================================
# EVIDENCE RANKING
# ==================================================

def rank_evidence(
    claim: Claim,
    evidence_list: List[Evidence],
    top_k: int = DEFAULT_TOP_K
) -> List[RankedEvidence]:
    """
    Rank evidence for a given claim.

    Returns explainable ranking results containing:

    - Original evidence
    - Rank
    - Normalized BM25 score
    - Semantic similarity score
    - Combined relevance score

    Ranking does not determine factual correctness.
    """

    # ----------------------------------------------
    # Input validation
    # ----------------------------------------------

    if not isinstance(claim, Claim):
        raise TypeError("claim must be a Claim object.")

    if not isinstance(evidence_list, list):
        raise TypeError("evidence_list must be a list.")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    if not evidence_list:
        return []

    for evidence in evidence_list:

        if not isinstance(evidence, Evidence):
            raise TypeError(
                "Every item must be an Evidence object."
            )

        if evidence.claim_id != claim.claim_id:
            raise ValueError(
                "Evidence claim_id does not match the supplied claim."
            )

    # ----------------------------------------------
    # Step 1: BM25 scoring
    # ----------------------------------------------

    bm25_scores = calculate_bm25_scores(
        claim.claim_text,
        evidence_list
    )

    normalized_bm25 = normalize_scores(
        bm25_scores
    )

    # ----------------------------------------------
    # Step 2: Semantic similarity
    # ----------------------------------------------

    semantic_scores = calculate_semantic_scores(
        claim.claim_text,
        evidence_list
    )

    # ----------------------------------------------
    # Step 3: Combined scoring
    # ----------------------------------------------

    combined_scores = calculate_combined_scores(
        bm25_scores,
        semantic_scores
    )

    # ----------------------------------------------
    # Step 4: Create ranking records
    # ----------------------------------------------

    ranking_records = []

    for evidence, bm25, semantic, combined in zip(
        evidence_list,
        normalized_bm25,
        semantic_scores,
        combined_scores
    ):

        ranking_records.append({
            "evidence": evidence,
            "bm25_score": bm25,
            "semantic_score": semantic,
            "combined_score": combined
        })

    # ----------------------------------------------
    # Step 5: Sort by combined relevance
    # ----------------------------------------------

    ranking_records.sort(
        key=lambda item: item["combined_score"],
        reverse=True
    )

    # ----------------------------------------------
    # Step 6: Select top-K and assign ranks
    # ----------------------------------------------

    ranked_results = []

    for index, record in enumerate(
        ranking_records[:top_k],
        start=1
    ):

        ranked_results.append(
            RankedEvidence(
                evidence=record["evidence"],
                rank=index,
                bm25_score=record["bm25_score"],
                semantic_score=record["semantic_score"],
                combined_score=record["combined_score"]
            )
        )

    logger.info(
        "Ranked %d evidence passages; selected top %d.",
        len(evidence_list),
        len(ranked_results)
    )

    return ranked_results
