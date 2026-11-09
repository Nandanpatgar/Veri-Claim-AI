
import pytest

from unittest.mock import Mock, patch

from backend.data_contracts import Claim, Evidence

from backend.evidence_retriever import (
    tokenize,
    is_relevant,
    search_wikipedia,
    fetch_wikipedia_extracts,
    extract_wikipedia_evidence,
    search_tavily,
    extract_tavily_evidence,
    normalize_url,
    deduplicate_evidence,
    retrieve_evidence_for_claim,
    retrieve_evidence
)


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture
def sample_claim():

    return Claim(
        claim_text="India gained independence in 1947.",
        original_text="India gained independence in 1947.",
        claim_type="historical"
    )


@pytest.fixture
def wikipedia_results():

    return [
        {
            "title": "Indian Independence",
            "snippet": "India became independent in 1947.",
            "pageid": 12345
        }
    ]


@pytest.fixture
def wikipedia_extracts():

    return {
        "12345": (
            "India gained independence from British rule "
            "on 15 August 1947."
        )
    }


@pytest.fixture
def tavily_results():

    return [
        {
            "title": "India Independence History",
            "content": (
                "India gained independence from British rule "
                "on 15 August 1947."
            ),
            "url": "https://example.com/india"
        }
    ]


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def test_tokenize():

    result = tokenize(
        "India gained independence in 1947."
    )

    assert "india" in result
    assert "independence" in result
    assert "1947" in result
    assert "in" not in result


def test_tokenize_empty_text():

    assert tokenize("") == set()


# --------------------------------------------------
# RELEVANCE
# --------------------------------------------------

def test_is_relevant_true():

    result = is_relevant(
        "India gained independence in 1947.",
        "India became independent in August 1947."
    )

    assert result is True


def test_is_relevant_false():

    result = is_relevant(
        "India gained independence in 1947.",
        "The weather in London is pleasant."
    )

    assert result is False


def test_is_relevant_empty_text():

    assert is_relevant("", "India independence") is False


# --------------------------------------------------
# WIKIPEDIA SEARCH
# --------------------------------------------------

@patch("backend.evidence_retriever.requests.get")
def test_search_wikipedia_success(
    mock_get,
    wikipedia_results
):

    mock_response = Mock()

    mock_response.json.return_value = {
        "query": {
            "search": wikipedia_results
        }
    }

    mock_get.return_value = mock_response

    result = search_wikipedia("India independence")

    assert result == wikipedia_results

    mock_response.raise_for_status.assert_called_once()


@patch("backend.evidence_retriever.requests.get")
def test_search_wikipedia_failure(mock_get):

    import requests

    mock_get.side_effect = requests.exceptions.RequestException(
        "Connection failed"
    )

    with pytest.raises(RuntimeError):

        search_wikipedia("India independence")


# --------------------------------------------------
# WIKIPEDIA ARTICLE EXTRACTION
# --------------------------------------------------

@patch("backend.evidence_retriever.requests.get")
def test_fetch_wikipedia_extracts_success(mock_get):

    mock_response = Mock()

    mock_response.json.return_value = {
        "query": {
            "pages": {
                "12345": {
                    "pageid": 12345,
                    "title": "Indian Independence",
                    "extract": (
                        "India gained independence in 1947."
                    )
                }
            }
        }
    }

    mock_get.return_value = mock_response

    results = [
        {
            "title": "Indian Independence",
            "pageid": 12345
        }
    ]

    result = fetch_wikipedia_extracts(results)

    assert "12345" in result
    assert "1947" in result["12345"]


def test_fetch_wikipedia_extracts_empty_results():

    result = fetch_wikipedia_extracts([])

    assert result == {}


@patch("backend.evidence_retriever.requests.get")
def test_fetch_wikipedia_extracts_failure(mock_get):

    import requests

    mock_get.side_effect = requests.exceptions.RequestException(
        "Wikipedia unavailable"
    )

    results = [
        {
            "title": "Indian Independence",
            "pageid": 12345
        }
    ]

    with pytest.raises(RuntimeError):

        fetch_wikipedia_extracts(results)


# --------------------------------------------------
# WIKIPEDIA EVIDENCE CONVERSION
# --------------------------------------------------

def test_extract_wikipedia_evidence(
    sample_claim,
    wikipedia_results,
    wikipedia_extracts
):

    result = extract_wikipedia_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        wikipedia_results,
        wikipedia_extracts
    )

    assert len(result) == 1

    assert isinstance(result[0], Evidence)

    assert result[0].claim_id == sample_claim.claim_id

    assert result[0].source_title == "Indian Independence"

    assert result[0].retrieval_provider == "Wikipedia"

    assert "1947" in result[0].passage

    assert result[0].source_url == (
        "https://en.wikipedia.org/wiki/Indian_Independence"
    )


def test_extract_wikipedia_empty_results(sample_claim):

    result = extract_wikipedia_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        [],
        {}
    )

    assert result == []


def test_extract_wikipedia_missing_extract(
    sample_claim,
    wikipedia_results
):

    result = extract_wikipedia_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        wikipedia_results,
        {}
    )

    assert result == []


def test_extract_wikipedia_irrelevant_result(sample_claim):

    results = [
        {
            "title": "London Weather",
            "pageid": 999
        }
    ]

    extracts = {
        "999": "London experiences rainfall throughout the year."
    }

    result = extract_wikipedia_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        results,
        extracts
    )

    assert result == []


# --------------------------------------------------
# TAVILY SEARCH
# --------------------------------------------------

@patch("backend.evidence_retriever.TavilyClient")
def test_search_tavily_success(
    mock_client,
    tavily_results
):

    mock_instance = Mock()

    mock_instance.search.return_value = {
        "results": tavily_results
    }

    mock_client.return_value = mock_instance

    with patch(
        "backend.evidence_retriever.TAVILY_API_KEY",
        "test-api-key"
    ):

        result = search_tavily("India independence")

    assert result == tavily_results


def test_search_tavily_missing_api_key():

    with patch(
        "backend.evidence_retriever.TAVILY_API_KEY",
        None
    ):

        with pytest.raises(RuntimeError):

            search_tavily("India independence")


@patch("backend.evidence_retriever.TavilyClient")
def test_search_tavily_failure(mock_client):

    mock_client.side_effect = Exception(
        "Tavily unavailable"
    )

    with patch(
        "backend.evidence_retriever.TAVILY_API_KEY",
        "test-api-key"
    ):

        with pytest.raises(RuntimeError):

            search_tavily("India independence")


# --------------------------------------------------
# TAVILY EVIDENCE CONVERSION
# --------------------------------------------------

def test_extract_tavily_evidence(
    sample_claim,
    tavily_results
):

    result = extract_tavily_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        tavily_results
    )

    assert len(result) == 1

    assert isinstance(result[0], Evidence)

    assert result[0].claim_id == sample_claim.claim_id

    assert result[0].retrieval_provider == "Tavily"

    assert result[0].source_url == (
        "https://example.com/india"
    )


def test_extract_tavily_empty_results(sample_claim):

    result = extract_tavily_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        []
    )

    assert result == []


def test_extract_tavily_irrelevant_result(sample_claim):

    results = [
        {
            "title": "London Weather",
            "content": "The weather in London is pleasant.",
            "url": "https://example.com/weather"
        }
    ]

    result = extract_tavily_evidence(
        sample_claim.claim_id,
        sample_claim.claim_text,
        results
    )

    assert result == []


# --------------------------------------------------
# URL NORMALIZATION
# --------------------------------------------------

def test_normalize_url():

    result = normalize_url(
        "https://www.example.com/article/?tracking=123"
    )

    assert result == "https://example.com/article"


def test_normalize_url_different_paths():

    first = normalize_url(
        "https://example.com/article-one"
    )

    second = normalize_url(
        "https://example.com/article-two"
    )

    assert first != second


# --------------------------------------------------
# DEDUPLICATION
# --------------------------------------------------

def test_deduplicate_evidence(sample_claim):

    evidence_one = Evidence(
        claim_id=sample_claim.claim_id,
        passage="India gained independence in 1947.",
        source_title="India Independence",
        source_url="https://example.com/india",
        source_type="webpage",
        retrieval_provider="Wikipedia"
    )

    evidence_two = Evidence(
        claim_id=sample_claim.claim_id,
        passage="India became independent in 1947.",
        source_title="India History",
        source_url="https://example.com/india/",
        source_type="webpage",
        retrieval_provider="Tavily"
    )

    result = deduplicate_evidence(
        [evidence_one, evidence_two]
    )

    assert len(result) == 1


def test_deduplicate_different_urls(sample_claim):

    evidence_one = Evidence(
        claim_id=sample_claim.claim_id,
        passage="Evidence one",
        source_title="Source one",
        source_url="https://example.com/one",
        source_type="webpage",
        retrieval_provider="Wikipedia"
    )

    evidence_two = Evidence(
        claim_id=sample_claim.claim_id,
        passage="Evidence two",
        source_title="Source two",
        source_url="https://example.com/two",
        source_type="webpage",
        retrieval_provider="Tavily"
    )

    result = deduplicate_evidence(
        [evidence_one, evidence_two]
    )

    assert len(result) == 2


# --------------------------------------------------
# SINGLE CLAIM RETRIEVAL
# --------------------------------------------------

@patch("backend.evidence_retriever.search_wikipedia")
@patch("backend.evidence_retriever.fetch_wikipedia_extracts")
@patch("backend.evidence_retriever.search_tavily")
def test_retrieve_evidence_for_claim(
    mock_tavily,
    mock_extracts,
    mock_wikipedia,
    sample_claim,
    wikipedia_results,
    wikipedia_extracts,
    tavily_results
):

    mock_wikipedia.return_value = wikipedia_results

    mock_extracts.return_value = wikipedia_extracts

    mock_tavily.return_value = tavily_results

    result = retrieve_evidence_for_claim(sample_claim)

    assert result["claim_id"] == sample_claim.claim_id

    assert len(result["evidence"]) == 2

    assert result["errors"] == {}

    assert result["provider_status"]["Wikipedia"]["status"] == "success"

    assert result["provider_status"]["Tavily"]["status"] == "success"


@patch("backend.evidence_retriever.search_wikipedia")
@patch("backend.evidence_retriever.search_tavily")
def test_wikipedia_failure_tavily_success(
    mock_tavily,
    mock_wikipedia,
    sample_claim,
    tavily_results
):

    mock_wikipedia.side_effect = RuntimeError(
        "Wikipedia unavailable"
    )

    mock_tavily.return_value = tavily_results

    result = retrieve_evidence_for_claim(sample_claim)

    assert len(result["evidence"]) == 1

    assert "Wikipedia" in result["errors"]

    assert result["provider_status"]["Tavily"]["status"] == "success"


@patch("backend.evidence_retriever.search_wikipedia")
@patch("backend.evidence_retriever.fetch_wikipedia_extracts")
@patch("backend.evidence_retriever.search_tavily")
def test_tavily_failure_wikipedia_success(
    mock_tavily,
    mock_extracts,
    mock_wikipedia,
    sample_claim,
    wikipedia_results,
    wikipedia_extracts
):

    mock_wikipedia.return_value = wikipedia_results

    mock_extracts.return_value = wikipedia_extracts

    mock_tavily.side_effect = RuntimeError(
        "Tavily unavailable"
    )

    result = retrieve_evidence_for_claim(sample_claim)

    assert len(result["evidence"]) == 1

    assert "Tavily" in result["errors"]

    assert result["provider_status"]["Wikipedia"]["status"] == "success"


@patch("backend.evidence_retriever.search_wikipedia")
@patch("backend.evidence_retriever.search_tavily")
def test_both_providers_fail(
    mock_tavily,
    mock_wikipedia,
    sample_claim
):

    mock_wikipedia.side_effect = RuntimeError(
        "Wikipedia unavailable"
    )

    mock_tavily.side_effect = RuntimeError(
        "Tavily unavailable"
    )

    result = retrieve_evidence_for_claim(sample_claim)

    assert result["evidence"] == []

    assert "Wikipedia" in result["errors"]

    assert "Tavily" in result["errors"]


@patch("backend.evidence_retriever.search_wikipedia")
@patch("backend.evidence_retriever.fetch_wikipedia_extracts")
@patch("backend.evidence_retriever.search_tavily")
def test_successful_search_with_no_results(
    mock_tavily,
    mock_extracts,
    mock_wikipedia,
    sample_claim
):

    mock_wikipedia.return_value = []

    mock_extracts.return_value = {}

    mock_tavily.return_value = []

    result = retrieve_evidence_for_claim(sample_claim)

    assert result["evidence"] == []

    assert result["errors"] == {}

    assert result["provider_status"]["Wikipedia"]["status"] == "success"

    assert result["provider_status"]["Tavily"]["status"] == "success"


# --------------------------------------------------
# MULTIPLE CLAIM RETRIEVAL
# --------------------------------------------------

@patch("backend.evidence_retriever.retrieve_evidence_for_claim")
def test_retrieve_evidence_success(
    mock_retrieve,
    sample_claim
):

    mock_retrieve.return_value = {
        "claim_id": sample_claim.claim_id,
        "evidence": [
            Mock(spec=Evidence)
        ],
        "errors": {},
        "provider_status": {}
    }

    result = retrieve_evidence(
        [sample_claim]
    )

    assert len(result["evidence"]) == 1

    assert sample_claim.claim_id in result["retrieval_report"]

    assert (
        result["retrieval_report"][sample_claim.claim_id]
        ["evidence_count"] == 1
    )


def test_retrieve_evidence_empty_list():

    result = retrieve_evidence([])

    assert result["evidence"] == []

    assert result["retrieval_report"] == {}


def test_retrieve_evidence_invalid_input():

    with pytest.raises(ValueError):

        retrieve_evidence("invalid")


def test_retrieve_evidence_invalid_claim():

    with pytest.raises(ValueError):

        retrieve_evidence([{"claim": "invalid"}])
