
"""
Tests for the NLI Verification Module.

Tests:
1. NLI label normalization
2. NLI prediction
3. Evidence verification
4. Multiple evidence verification
5. Batch claim verification
6. Input validation
"""

import pytest

from unittest.mock import patch

from backend.data_contracts import (
    Claim,
    Evidence,
    EvidenceVerification
)

from backend.evidence_ranker import RankedEvidence

from backend import nli_verifier


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

    return Evidence(
        claim_id=sample_claim.claim_id,
        passage="India became independent on 15 August 1947.",
        source_title="Indian Independence",
        source_url="https://example.com/india",
        source_type="encyclopedia",
        retrieval_provider="test"
    )


@pytest.fixture
def sample_ranked_evidence(sample_evidence):

    return RankedEvidence(
        evidence=sample_evidence,
        rank=1,
        bm25_score=0.9,
        semantic_score=0.8,
        combined_score=0.84
    )


@pytest.fixture
def sample_prediction():

    return {
        "nli_label": "ENTAILMENT",
        "nli_confidence": 0.95,
        "entailment_score": 0.95,
        "contradiction_score": 0.02,
        "neutral_score": 0.03
    }


# ==================================================
# LABEL NORMALIZATION TESTS
# ==================================================

def test_normalize_entailment_label():

    assert (
        nli_verifier.normalize_nli_label("entailment")
        == "ENTAILMENT"
    )


def test_normalize_contradiction_label():

    assert (
        nli_verifier.normalize_nli_label("contradiction")
        == "CONTRADICTION"
    )


def test_normalize_neutral_label():

    assert (
        nli_verifier.normalize_nli_label("neutral")
        == "NEUTRAL"
    )


def test_normalize_invalid_label():

    with pytest.raises(ValueError):

        nli_verifier.normalize_nli_label("unknown")


def test_normalize_non_string_label():

    with pytest.raises(ValueError):

        nli_verifier.normalize_nli_label(123)


# ==================================================
# PREDICTION TESTS
# ==================================================

def test_predict_nli_entailment(sample_prediction):

    with patch(
        "backend.nli_verifier.get_nli_model"
    ) as mock_model:

        mock_tokenizer = patch(
            "backend.nli_verifier._nli_tokenizer"
        )

        with mock_tokenizer:

            nli_verifier._device = "cpu"

            nli_verifier._nli_label_mapping = {
                0: "CONTRADICTION",
                1: "ENTAILMENT",
                2: "NEUTRAL"
            }

            import torch

            class MockTokenizer:

                def __call__(self, *args, **kwargs):

                    return {
                        "input_ids": torch.tensor([[1, 2, 3]])
                    }

            class MockModel:

                def __call__(self, **kwargs):

                    class Output:

                        logits = torch.tensor(
                            [[-2.0, 4.0, -1.0]]
                        )

                    return Output()

            mock_model.return_value = (
                MockTokenizer(),
                MockModel()
            )

            result = nli_verifier.predict_nli(
                "India became independent in 1947.",
                "India gained independence in 1947."
            )

    assert result["nli_label"] == "ENTAILMENT"
    assert result["entailment_score"] > result["contradiction_score"]


def test_predict_nli_contradiction():

    with patch(
        "backend.nli_verifier.get_nli_model"
    ) as mock_model:

        import torch

        nli_verifier._device = "cpu"

        nli_verifier._nli_label_mapping = {
            0: "CONTRADICTION",
            1: "ENTAILMENT",
            2: "NEUTRAL"
        }

        class MockTokenizer:

            def __call__(self, *args, **kwargs):

                return {
                    "input_ids": torch.tensor([[1, 2, 3]])
                }

        class MockModel:

            def __call__(self, **kwargs):

                class Output:

                    logits = torch.tensor(
                        [[5.0, -2.0, -1.0]]
                    )

                return Output()

        mock_model.return_value = (
            MockTokenizer(),
            MockModel()
        )

        result = nli_verifier.predict_nli(
            "India gained independence in 1947.",
            "India gained independence in 1950."
        )

    assert result["nli_label"] == "CONTRADICTION"


def test_predict_nli_neutral():

    with patch(
        "backend.nli_verifier.get_nli_model"
    ) as mock_model:

        import torch

        nli_verifier._device = "cpu"

        nli_verifier._nli_label_mapping = {
            0: "CONTRADICTION",
            1: "ENTAILMENT",
            2: "NEUTRAL"
        }

        class MockTokenizer:

            def __call__(self, *args, **kwargs):

                return {
                    "input_ids": torch.tensor([[1, 2, 3]])
                }

        class MockModel:

            def __call__(self, **kwargs):

                class Output:

                    logits = torch.tensor(
                        [[-1.0, -2.0, 5.0]]
                    )

                return Output()

        mock_model.return_value = (
            MockTokenizer(),
            MockModel()
        )

        result = nli_verifier.predict_nli(
            "India gained independence in 1947.",
            "India has a large population."
        )

    assert result["nli_label"] == "NEUTRAL"


def test_predict_nli_empty_premise():

    with pytest.raises(ValueError):

        nli_verifier.predict_nli(
            "",
            "India gained independence in 1947."
        )


def test_predict_nli_empty_hypothesis():

    with pytest.raises(ValueError):

        nli_verifier.predict_nli(
            "India became independent in 1947.",
            ""
        )


def test_predict_nli_non_string_premise():

    with pytest.raises(ValueError):

        nli_verifier.predict_nli(
            123,
            "India gained independence in 1947."
        )


# ==================================================
# EVIDENCE VERIFICATION TESTS
# ==================================================

def test_verify_evidence_success(
    sample_claim,
    sample_ranked_evidence,
    sample_prediction
):

    with patch(
        "backend.nli_verifier.predict_nli",
        return_value=sample_prediction
    ):

        result = nli_verifier.verify_evidence(
            sample_claim,
            sample_ranked_evidence
        )

    assert isinstance(result, EvidenceVerification)

    assert result.claim_id == sample_claim.claim_id

    assert result.evidence_id == (
        sample_ranked_evidence.evidence.evidence_id
    )

    assert result.nli_label == "ENTAILMENT"

    assert result.nli_confidence == 0.95


def test_verify_evidence_invalid_claim(
    sample_ranked_evidence
):

    with pytest.raises(TypeError):

        nli_verifier.verify_evidence(
            "invalid claim",
            sample_ranked_evidence
        )


def test_verify_evidence_invalid_ranked_evidence(
    sample_claim
):

    with pytest.raises(TypeError):

        nli_verifier.verify_evidence(
            sample_claim,
            "invalid evidence"
        )


def test_verify_evidence_mismatched_claim_id(
    sample_claim,
    sample_ranked_evidence
):

    another_claim = Claim(
        claim_text="Another claim.",
        original_text="Another claim.",
        claim_type="general"
    )

    with pytest.raises(ValueError):

        nli_verifier.verify_evidence(
            another_claim,
            sample_ranked_evidence
        )


# ==================================================
# MULTIPLE EVIDENCE VERIFICATION TESTS
# ==================================================

def test_verify_claim_multiple_evidence(
    sample_claim,
    sample_ranked_evidence,
    sample_prediction
):

    second_evidence = Evidence(
        claim_id=sample_claim.claim_id,
        passage="India became independent from Britain in 1947.",
        source_title="History",
        source_url="https://example.com/history",
        source_type="encyclopedia",
        retrieval_provider="test"
    )

    second_ranked = RankedEvidence(
        evidence=second_evidence,
        rank=2,
        bm25_score=0.7,
        semantic_score=0.75,
        combined_score=0.73
    )

    with patch(
        "backend.nli_verifier.predict_nli",
        return_value=sample_prediction
    ) as mock_predict:

        results = nli_verifier.verify_claim(
            sample_claim,
            [
                sample_ranked_evidence,
                second_ranked
            ]
        )

    assert len(results) == 2

    assert all(
        isinstance(result, EvidenceVerification)
        for result in results
    )

    assert mock_predict.call_count == 2


def test_verify_claim_empty_evidence(sample_claim):

    results = nli_verifier.verify_claim(
        sample_claim,
        []
    )

    assert results == []


def test_verify_claim_invalid_input():

    with pytest.raises(TypeError):

        nli_verifier.verify_claim(
            "invalid claim",
            []
        )


def test_verify_claim_non_list_evidence(sample_claim):

    with pytest.raises(TypeError):

        nli_verifier.verify_claim(
            sample_claim,
            "invalid evidence"
        )


# ==================================================
# BATCH VERIFICATION TESTS
# ==================================================

def test_verify_claims_success(
    sample_claim,
    sample_ranked_evidence,
    sample_prediction
):

    with patch(
        "backend.nli_verifier.verify_claim",
        return_value=[
            EvidenceVerification(
                claim_id=sample_claim.claim_id,
                evidence_id=sample_ranked_evidence.evidence.evidence_id,
                nli_label="ENTAILMENT",
                nli_confidence=0.95,
                entailment_score=0.95,
                contradiction_score=0.02,
                neutral_score=0.03
            )
        ]
    ):

        results = nli_verifier.verify_claims(
            [sample_claim],
            {
                sample_claim.claim_id: [
                    sample_ranked_evidence
                ]
            }
        )

    assert sample_claim.claim_id in results

    assert len(results[sample_claim.claim_id]) == 1


def test_verify_claims_empty_claims():

    results = nli_verifier.verify_claims(
        [],
        {}
    )

    assert results == {}


def test_verify_claims_invalid_claims():

    with pytest.raises(TypeError):

        nli_verifier.verify_claims(
            ["invalid claim"],
            {}
        )


def test_verify_claims_invalid_mapping():

    with pytest.raises(TypeError):

        nli_verifier.verify_claims(
            [],
            []
        )
