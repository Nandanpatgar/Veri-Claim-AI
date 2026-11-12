"""
NLI Diagnostic Evaluation

Tests DeBERTa against different types of
claim-evidence relationships.
"""

from backend.nli_verifier import predict_nli


test_cases = [

    # 1. Correct relationship
    {
        "name": "Correct Relationship",
        "premise": "Mumbai is the capital of Maharashtra.",
        "hypothesis": "Mumbai is Maharashtra's capital.",
        "expected": "ENTAILMENT"
    },

    # 2. Incorrect subject
    {
        "name": "Incorrect Subject",
        "premise": "Mumbai is the capital of Maharashtra.",
        "hypothesis": "Pune is the capital of Maharashtra.",
        "expected": "CONTRADICTION"
    },

    # 3. Incorrect relationship / entity mismatch
    {
        "name": "Incorrect Geographic Relationship",
        "premise": "Mumbai is the capital of Maharashtra.",
        "hypothesis": "The capital of India is Mumbai.",
        "expected": "NEUTRAL"
    },

    # 4. Incorrect object
    {
        "name": "Incorrect Object",
        "premise": "Mumbai is the capital of Maharashtra.",
        "hypothesis": "Mumbai is the capital of Karnataka.",
        "expected": "CONTRADICTION"
    },

    # 5. Equivalent wording
    {
        "name": "Equivalent Wording",
        "premise": "India became independent from British rule on 15 August 1947.",
        "hypothesis": "British rule in India ended on 15 August 1947.",
        "expected": "ENTAILMENT"
    },

    # 6. Multiple statements
    {
        "name": "Multiple Statements",
        "premise": (
            "Mumbai is the capital of Maharashtra. "
            "It is also India's financial capital."
        ),
        "hypothesis": "Mumbai is the capital of India.",
        "expected": "NEUTRAL"
    }
]


passed = 0
failed = 0


for case in test_cases:

    result = predict_nli(
        premise=case["premise"],
        hypothesis=case["hypothesis"]
    )

    predicted = result["nli_label"]

    is_pass = predicted == case["expected"]

    if is_pass:
        passed += 1
    else:
        failed += 1

    print("\n" + "=" * 65)

    print("Scenario:", case["name"])

    print("Premise:", case["premise"])

    print("Hypothesis:", case["hypothesis"])

    print("Expected:", case["expected"])

    print("Predicted:", predicted)

    print("Confidence:", round(
        result["nli_confidence"], 4
    ))

    print("Entailment:", round(
        result["entailment_score"], 4
    ))

    print("Contradiction:", round(
        result["contradiction_score"], 4
    ))

    print("Neutral:", round(
        result["neutral_score"], 4
    ))

    print("Result:", "PASS" if is_pass else "REVIEW")


print("\n" + "=" * 65)

print("NLI DIAGNOSTIC SUMMARY")

print("Total scenarios:", len(test_cases))

print("Passed:", passed)

print("Failed:", failed)

print(
    "Overall:",
    "PASS" if failed == 0 else "REVIEW REQUIRED"
)

print("=" * 65)