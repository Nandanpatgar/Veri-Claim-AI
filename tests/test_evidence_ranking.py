
import pytest
from unittest.mock import patch

from backend.data_contracts import Claim, Evidence
from backend.evidence_ranker import (
    RankedEvidence,
    tokenize,
    calculate_bm25_scores,
    normalize_scores,
    calculate_combined_scores,
    calculate_semantic_scores,
    rank_evidence,
)


# ==================================================
# FIXTURES
# ==================================================

@pytest.fixture
def sample_claim():
    return Claim(
        claim_text="India gained independence in 1947.",
        original_text="India gained independence in 1947.",
        claim_type="historical"
    )


@pytest.fixture
def sample_evidence(sample_claim):
    return [
        Evidence(
            claim_id=sample_claim.claim_id,
            passage="India gained independence from British rule on 15 August 1947.",
            source_title="Indian Independence",
            source_url="https://example.com/india",
            source_type="encyclopedia",
            retrieval_provider="wikipedia"
        ),
        Evidence(
            claim_id=sample_claim.claim_id,
            passage="India became a republic on 26 January 1950.",
            source_title="Republic of India",
            source_url="https://example.com/republic",
            source_type="encyclopedia",
            retrieval_provider="wikipedia"
        ),
        Evidence(
            claim_id=sample_claim.claim_id,
            passage="The Indian independence movement opposed British rule.",
            source_title="Independence Movement",
            source_url="https://example.com/movement",
            source_type="article",
            retrieval_provider="tavily"
        )
    ]


# ==================================================
# 1. TOKENIZATION TESTS
# ==================================================

def test_tokenize_lowercase():
    assert tokenize("INDIA Gained Freedom") == [
        "india", "gained", "freedom"
    ]


def test_tokenize_removes_punctuation():
    assert tokenize("India, gained freedom!") == [
        "india", "gained", "freedom"
    ]


def test_tokenize_empty_text():
    assert tokenize("") == []


def test_tokenize_invalid_input():
    with pytest.raises(ValueError):
        tokenize(None)


# ==================================================
# 2. BM25 TESTS
# ==================================================

def test_bm25_returns_one_score_per_evidence(
    sample_claim,
    sample_evidence
):
    scores = calculate_bm25_scores(
        sample_claim.claim_text,
        sample_evidence
    )

    assert len(scores) == len(sample_evidence)


def test_bm25_empty_evidence(sample_claim):
    assert calculate_bm25_scores(
        sample_claim.claim_text,
        []
    ) == []


def test_bm25_empty_claim(sample_evidence):
    assert calculate_bm25_scores(
        "",
        sample_evidence
    ) == [0.0, 0.0, 0.0]


def test_bm25_relevant_evidence_gets_score(
    sample_claim,
    sample_evidence
):
    scores = calculate_bm25_scores(
        sample_claim.claim_text,
        sample_evidence
    )

    assert scores[0] > scores[1]


# ==================================================
# 3. NORMALIZATION TESTS
# ==================================================

def test_normalize_scores():
    assert normalize_scores([2, 4, 6]) == [
        0.0, 0.5, 1.0
    ]


def test_normalize_identical_scores():
    assert normalize_scores([5, 5, 5]) == [
        0.0, 0.0, 0.0
    ]


def test_normalize_empty_scores():
    assert normalize_scores([]) == []


def test_normalize_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        normalize_scores([1.0, float("nan")])


# ==================================================
# 4. COMBINED SCORING TESTS
# ==================================================

def test_combined_scores():
    result = calculate_combined_scores(
        [0, 5, 10],
        [0.2, 0.5, 0.9]
    )

    assert result[0] == pytest.approx(0.12)
    assert result[1] == pytest.approx(0.50)
    assert result[2] == pytest.approx(0.94)


def test_combined_scores_length_mismatch():
    with pytest.raises(ValueError):
        calculate_combined_scores(
            [1, 2],
            [0.5]
        )


def test_combined_scores_empty():
    assert calculate_combined_scores([], []) == []


def test_combined_scores_rejects_non_finite_semantic_scores():
    with pytest.raises(ValueError):
        calculate_combined_scores(
            [1, 2],
            [0.5, float("nan")]
        )


# ==================================================
# 5. SEMANTIC SIMILARITY TESTS
# ==================================================

@patch("backend.evidence_ranker.get_embedding_model")
def test_semantic_scores(
    mock_get_model,
    sample_claim,
    sample_evidence
):
    import torch

    class MockModel:

        def encode(self, texts, **kwargs):

            if isinstance(texts, str):
                return torch.tensor([1.0, 0.0])

            return torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.6]
            ])

    mock_get_model.return_value = MockModel()

    scores = calculate_semantic_scores(
        sample_claim.claim_text,
        sample_evidence
    )

    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.8)


def test_semantic_scores_empty(sample_claim):
    assert calculate_semantic_scores(
        sample_claim.claim_text,
        []
    ) == []


# ==================================================
# 6. EVIDENCE RANKING TESTS
# ==================================================

@patch("backend.evidence_ranker.calculate_semantic_scores")
def test_rank_evidence_returns_ranked_objects(
    mock_semantic,
    sample_claim,
    sample_evidence
):
    mock_semantic.return_value = [0.9, 0.2, 0.5]

    ranked = rank_evidence(
        sample_claim,
        sample_evidence
    )

    assert len(ranked) == 3

    assert all(
        isinstance(item, RankedEvidence)
        for item in ranked
    )


@patch("backend.evidence_ranker.calculate_semantic_scores")
def test_rank_evidence_descending_order(
    mock_semantic,
    sample_claim,
    sample_evidence
):
    mock_semantic.return_value = [0.9, 0.2, 0.5]

    ranked = rank_evidence(
        sample_claim,
        sample_evidence
    )

    assert ranked[0].evidence == sample_evidence[0]
    assert ranked[0].combined_score >= ranked[1].combined_score
    assert ranked[1].combined_score >= ranked[2].combined_score


@patch("backend.evidence_ranker.calculate_semantic_scores")
def test_rank_evidence_assigns_correct_ranks(
    mock_semantic,
    sample_claim,
    sample_evidence
):
    mock_semantic.return_value = [0.9, 0.2, 0.5]

    ranked = rank_evidence(
        sample_claim,
        sample_evidence
    )

    assert [item.rank for item in ranked] == [1, 2, 3]


@patch("backend.evidence_ranker.calculate_semantic_scores")
def test_rank_evidence_stores_individual_scores(
    mock_semantic,
    sample_claim,
    sample_evidence
):
    mock_semantic.return_value = [0.9, 0.2, 0.5]

    ranked = rank_evidence(
        sample_claim,
        sample_evidence
    )

    for item in ranked:
        assert 0.0 <= item.bm25_score <= 1.0
        assert -1.0 <= item.semantic_score <= 1.0
        assert item.combined_score == pytest.approx(
            0.4 * item.bm25_score +
            0.6 * item.semantic_score
        )


@patch("backend.evidence_ranker.calculate_semantic_scores")
def test_rank_evidence_top_k(
    mock_semantic,
    sample_claim,
    sample_evidence
):
    mock_semantic.return_value = [0.9, 0.2, 0.5]

    ranked = rank_evidence(
        sample_claim,
        sample_evidence,
        top_k=2
    )

    assert len(ranked) == 2
    assert [item.rank for item in ranked] == [1, 2]


def test_rank_evidence_empty_list(sample_claim):
    assert rank_evidence(sample_claim, []) == []


def test_rank_evidence_invalid_claim():
    with pytest.raises(TypeError):
        rank_evidence(
            "India gained independence in 1947.",
            []
        )


def test_rank_evidence_invalid_top_k(
    sample_claim,
    sample_evidence
):
    with pytest.raises(ValueError):
        rank_evidence(
            sample_claim,
            sample_evidence,
            top_k=0
        )


def test_rank_evidence_rejects_boolean_top_k(
    sample_claim,
    sample_evidence
):
    with pytest.raises(ValueError):
        rank_evidence(
            sample_claim,
            sample_evidence,
            top_k=True
        )


def test_rank_evidence_rejects_mismatched_claim_id(
    sample_evidence
):
    another_claim = Claim(
        claim_text="The Earth revolves around the Sun.",
        original_text="The Earth revolves around the Sun.",
        claim_type="scientific"
    )

    with pytest.raises(ValueError):
        rank_evidence(
            another_claim,
            sample_evidence
        )


def test_rank_evidence_rejects_invalid_evidence(
    sample_claim
):
    with pytest.raises(TypeError):
        rank_evidence(
            sample_claim,
            ["Not an Evidence object"]
        )
