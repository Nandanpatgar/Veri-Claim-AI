# =====================================================
# VERI-CLAIM-AI 2.0
# CLAIM EXTRACTION MODULE
# =====================================================

import json
import requests

from backend.config import (
    OLLAMA_GENERATE_URL,
    CLAIM_EXTRACTION_MODEL,
    OLLAMA_TIMEOUT
)

from backend.data_contracts import Claim


# =====================================================
# 1. BUILD CLAIM EXTRACTION PROMPT
# =====================================================

def build_extraction_prompt(text):

    return f"""
You are an advanced factual claim extraction system.

TASK:
Extract ALL independently verifiable factual claims
from the provided text.

RULES:

1. Extract only factual statements.
2. Split compound sentences into atomic claims.
3. Each claim must express exactly one independently
   verifiable fact.
4. Preserve the original meaning.
5. Do not omit factual information.
6. Ignore opinions, greetings, questions and instructions.
7. Do not determine whether claims are true or false.
8. Never invent or assume information.
9. Avoid duplicate claims.
10. Assign a suitable claim_type to every claim.

IMPORTANT OUTPUT RULES:

- Return a JSON array.
- Even if there is only ONE claim, return an array
  containing one object.
- If there are no factual claims, return [].
- Do not wrap the array inside another object.
- Do not include explanations or Markdown.

REQUIRED OUTPUT FORMAT:

[
    {{
        "claim_text": "India gained independence in 1947.",
        "claim_type": "historical"
    }},
    {{
        "claim_text": "India became a republic in 1950.",
        "claim_type": "historical"
    }}
]

INPUT TEXT:
{text}
"""


# =====================================================
# 2. COMMUNICATE WITH OLLAMA
# =====================================================

def call_ollama(prompt):

    # Enforce a JSON array of claim objects.
    output_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "claim_text": {
                    "type": "string"
                },
                "claim_type": {
                    "type": "string"
                }
            },
            "required": [
                "claim_text",
                "claim_type"
            ],
            "additionalProperties": False
        }
    }

    payload = {
        "model": CLAIM_EXTRACTION_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": output_schema
    }

    try:

        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        llm_response = data.get("response", "")

        if not isinstance(llm_response, str) or not llm_response.strip():

            raise ValueError(
                "Ollama returned an empty response."
            )

        return llm_response

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Ollama communication failed: {error}"
        ) from error


# =====================================================
# 3. PARSE LLM RESPONSE
# =====================================================

def parse_llm_response(response):

    if not isinstance(response, str) or not response.strip():

        raise ValueError(
            "LLM response cannot be empty."
        )

    try:

        parsed_response = json.loads(response)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Invalid JSON response: {error}"
        ) from error

    if not isinstance(parsed_response, list):

        raise ValueError(
            "Expected a JSON array."
        )

    return parsed_response


# =====================================================
# 4. VALIDATE EXTRACTED CLAIMS
# =====================================================

def validate_extracted_claims(claims):

    if not isinstance(claims, list):

        raise ValueError(
            "Claims must be a list."
        )

    validated_claims = []

    for item in claims:

        if not isinstance(item, dict):

            raise ValueError(
                "Each claim must be a JSON object."
            )

        if set(item.keys()) != {
            "claim_text",
            "claim_type"
        }:

            raise ValueError(
                "Each claim must contain exactly "
                "claim_text and claim_type."
            )

        claim_text = item["claim_text"]

        claim_type = item["claim_type"]

        if (
            not isinstance(claim_text, str)
            or not claim_text.strip()
        ):

            raise ValueError(
                "Claim text cannot be empty."
            )

        if (
            not isinstance(claim_type, str)
            or not claim_type.strip()
        ):

            raise ValueError(
                "Claim type cannot be empty."
            )

        validated_claims.append(
            {
                "claim_text": claim_text.strip(),
                "claim_type": claim_type.strip().lower()
            }
        )

    return validated_claims


# =====================================================
# 5. MAIN CLAIM EXTRACTION FUNCTION
# =====================================================

def extract_claims(text):

    # Validate input.
    if not isinstance(text, str):

        raise ValueError(
            "Input must be a string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Input text cannot be empty."
        )

    # Step 1: Build extraction prompt.
    prompt = build_extraction_prompt(text)

    # Step 2: Send prompt to Llama through Ollama.
    llm_response = call_ollama(prompt)

    # Step 3: Parse JSON response.
    parsed_response = parse_llm_response(llm_response)

    # Step 4: Validate extracted claims.
    validated_claims = validate_extracted_claims(
        parsed_response
    )

    # Step 5: Convert dictionaries into Claim objects.
    claim_objects = []

    for item in validated_claims:

        claim = Claim(
            claim_text=item["claim_text"],
            original_text=text,
            claim_type=item["claim_type"]
        )

        claim_objects.append(claim)

    return claim_objects