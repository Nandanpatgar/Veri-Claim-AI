# ==========================================
# IMPORTS
# ==========================================

import pytest
import requests

from unittest.mock import patch, Mock

from backend.claim_extractor import (
    build_extraction_prompt,
    call_ollama,
    parse_llm_response,
    validate_extracted_claims,
    extract_claims
)

from backend.data_contracts import Claim


# ==========================================
# PROMPT GENERATION TESTS
# ==========================================

def test_build_extraction_prompt():

    prompt = build_extraction_prompt(
        "India gained independence in 1947."
    )

    assert "independence in 1947" in prompt
    assert "JSON array" in prompt
    assert "atomic claims" in prompt


def test_prompt_contains_input_text():

    text = "India became a republic in 1950."

    prompt = build_extraction_prompt(text)

    assert text in prompt


# ==========================================
# JSON PARSING TESTS
# ==========================================

def test_parse_valid_json():

    response = """
    [
        {
            "claim_text": "India gained independence in 1947.",
            "claim_type": "historical"
        }
    ]
    """

    result = parse_llm_response(response)

    assert isinstance(result, list)
    assert len(result) == 1


def test_parse_empty_response():

    with pytest.raises(ValueError):
        parse_llm_response("")


def test_parse_whitespace_response():

    with pytest.raises(ValueError):
        parse_llm_response("     ")


def test_parse_invalid_json():

    with pytest.raises(ValueError):
        parse_llm_response("This is not JSON")


def test_parse_non_array_json():

    with pytest.raises(ValueError):
        parse_llm_response(
            '{"claim_text": "India gained independence in 1947."}'
        )


def test_parse_empty_array():

    result = parse_llm_response("[]")

    assert result == []


def test_parse_non_string_response():

    with pytest.raises(ValueError):
        parse_llm_response(123)


# ==========================================
# CLAIM VALIDATION TESTS
# ==========================================

def test_validate_valid_claims():

    claims = [
        {
            "claim_text": "India gained independence in 1947.",
            "claim_type": "HISTORICAL"
        }
    ]

    result = validate_extracted_claims(claims)

    assert result[0]["claim_text"] == (
        "India gained independence in 1947."
    )

    assert result[0]["claim_type"] == "historical"


def test_validate_empty_claim_list():

    result = validate_extracted_claims([])

    assert result == []


def test_validate_invalid_claim_type():

    claims = [
        {
            "claim_text": "India gained independence in 1947.",
            "claim_type": ""
        }
    ]

    with pytest.raises(ValueError):
        validate_extracted_claims(claims)


def test_validate_empty_claim_text():

    claims = [
        {
            "claim_text": "",
            "claim_type": "historical"
        }
    ]

    with pytest.raises(ValueError):
        validate_extracted_claims(claims)


def test_validate_missing_fields():

    claims = [
        {
            "claim_text": "India gained independence in 1947."
        }
    ]

    with pytest.raises(ValueError):
        validate_extracted_claims(claims)


def test_validate_extra_fields():

    claims = [
        {
            "claim_text": "India gained independence in 1947.",
            "claim_type": "historical",
            "extra": "unexpected"
        }
    ]

    with pytest.raises(ValueError):
        validate_extracted_claims(claims)


def test_validate_non_dictionary_claim():

    with pytest.raises(ValueError):
        validate_extracted_claims(["invalid claim"])


def test_validate_non_string_claim_text():

    claims = [
        {
            "claim_text": 123,
            "claim_type": "historical"
        }
    ]

    with pytest.raises(ValueError):
        validate_extracted_claims(claims)


def test_validate_non_string_claim_type():

    claims = [
        {
            "claim_text": "India gained independence in 1947.",
            "claim_type": 123
        }
    ]

    with pytest.raises(ValueError):
        validate_extracted_claims(claims)


# ==========================================
# OLLAMA COMMUNICATION TESTS
# ==========================================

@patch("backend.claim_extractor.requests.post")
def test_call_ollama_success(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "response": (
            '[{"claim_text": "India gained independence in 1947.", '
            '"claim_type": "historical"}]'
        )
    }

    mock_post.return_value = mock_response

    result = call_ollama("Sample prompt")

    assert isinstance(result, str)
    assert "independence in 1947" in result

    mock_post.assert_called_once()


@patch("backend.claim_extractor.requests.post")
def test_call_ollama_connection_failure(mock_post):

    mock_post.side_effect = requests.exceptions.ConnectionError(
        "Ollama is unavailable"
    )

    with pytest.raises(RuntimeError):
        call_ollama("Sample prompt")


@patch("backend.claim_extractor.requests.post")
def test_call_ollama_http_failure(mock_post):

    mock_response = Mock()

    mock_response.raise_for_status.side_effect = (
        requests.exceptions.HTTPError("Server error")
    )

    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError):
        call_ollama("Sample prompt")


@patch("backend.claim_extractor.requests.post")
def test_call_ollama_empty_response(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "response": ""
    }

    mock_post.return_value = mock_response

    with pytest.raises(ValueError):
        call_ollama("Sample prompt")


# ==========================================
# MAIN EXTRACTION TESTS
# ==========================================

@patch("backend.claim_extractor.call_ollama")
def test_extract_claims_success(mock_ollama):

    mock_ollama.return_value = """
    [
        {
            "claim_text": "India gained independence in 1947.",
            "claim_type": "historical"
        },
        {
            "claim_text": "India became a republic in 1950.",
            "claim_type": "historical"
        }
    ]
    """

    original_text = (
        "India gained independence in 1947 "
        "and became a republic in 1950."
    )

    result = extract_claims(original_text)

    assert len(result) == 2

    assert isinstance(result[0], Claim)

    assert result[0].claim_text == (
        "India gained independence in 1947."
    )

    assert result[1].claim_text == (
        "India became a republic in 1950."
    )

    assert result[0].original_text == original_text
    assert result[1].original_text == original_text


@patch("backend.claim_extractor.call_ollama")
def test_extract_claims_no_factual_claims(mock_ollama):

    mock_ollama.return_value = "[]"

    result = extract_claims(
        "I think this is wonderful!"
    )

    assert result == []


def test_extract_claims_empty_input():

    with pytest.raises(ValueError):
        extract_claims("")


def test_extract_claims_whitespace_input():

    with pytest.raises(ValueError):
        extract_claims("     ")


def test_extract_claims_non_string_input():

    with pytest.raises(ValueError):
        extract_claims(123)


@patch("backend.claim_extractor.call_ollama")
def test_extract_claims_invalid_llm_response(mock_ollama):

    mock_ollama.return_value = "Invalid JSON"

    with pytest.raises(ValueError):
        extract_claims(
            "India gained independence in 1947."
        )


@patch("backend.claim_extractor.call_ollama")
def test_extract_claims_invalid_claim_structure(mock_ollama):

    mock_ollama.return_value = """
    [
        {
            "claim_text": "India gained independence in 1947."
        }
    ]
    """

    with pytest.raises(ValueError):
        extract_claims(
            "India gained independence in 1947."
        )