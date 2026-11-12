"""
Tests for the Veri-Claim-AI Flask API.
"""

import pytest
from unittest.mock import Mock

from backend.app import create_app
from backend.verification_service import VerificationPipelineError


# ==================================================
# FIXTURE
# ==================================================

@pytest.fixture
def client():

    mock_service = Mock()

    app = create_app(
        verification_service=mock_service
    )

    app.config["TESTING"] = True

    with app.test_client() as test_client:

        yield test_client, mock_service


# ==================================================
# 1. HEALTH CHECK TESTS
# ==================================================

def test_health_check(client):

    test_client, mock_service = client

    response = test_client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "success"

    assert data["service"] == "Veri-Claim-AI"

    mock_service.verify.assert_not_called()


# ==================================================
# 2. REQUEST VALIDATION TESTS
# ==================================================

def test_verify_without_json(client):

    test_client, mock_service = client

    response = test_client.post(
        "/api/verify",
        data="invalid request"
    )

    assert response.status_code == 400

    mock_service.verify.assert_not_called()


def test_verify_with_empty_json(client):

    test_client, mock_service = client

    response = test_client.post(
        "/api/verify",
        json={}
    )

    assert response.status_code == 400

    mock_service.verify.assert_not_called()


def test_verify_with_null_text(client):

    test_client, mock_service = client

    response = test_client.post(
        "/api/verify",
        json={
            "text": None
        }
    )

    assert response.status_code == 400

    mock_service.verify.assert_not_called()


def test_verify_with_invalid_json_body(client):

    test_client, mock_service = client

    response = test_client.post(
        "/api/verify",
        json=[]
    )

    assert response.status_code == 400

    mock_service.verify.assert_not_called()


# ==================================================
# 3. SUCCESSFUL VERIFICATION TEST
# ==================================================

def test_verify_success(client):

    test_client, mock_service = client

    mock_service.verify.return_value = {
        "pipeline_status": "completed",
        "report": {
            "results": []
        }
    }

    response = test_client.post(
        "/api/verify",
        json={
            "text": "India became independent in 1947."
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "success"

    assert data["data"]["pipeline_status"] == "completed"

    mock_service.verify.assert_called_once_with(
        "India became independent in 1947."
    )


# ==================================================
# 4. PIPELINE ERROR TEST
# ==================================================

def test_verify_pipeline_error(client):

    test_client, mock_service = client

    mock_service.verify.side_effect = (
        VerificationPipelineError(
            "claim_extraction",
            "LLM unavailable"
        )
    )

    response = test_client.post(
        "/api/verify",
        json={
            "text": "Test input"
        }
    )

    assert response.status_code == 422

    data = response.get_json()

    assert data["status"] == "error"

    assert data["stage"] == "claim_extraction"

    assert data["message"] == "LLM unavailable"


# ==================================================
# 5. UNEXPECTED ERROR TEST
# ==================================================

def test_verify_unexpected_error(client):

    test_client, mock_service = client

    mock_service.verify.side_effect = RuntimeError(
        "Unexpected error"
    )

    response = test_client.post(
        "/api/verify",
        json={
            "text": "Test input"
        }
    )

    assert response.status_code == 500

    data = response.get_json()

    assert data["status"] == "error"

    assert data["message"] == (
        "An unexpected server error occurred."
    )