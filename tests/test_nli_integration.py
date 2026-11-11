from backend.nli_verifier import predict_nli


test_cases = [
    {
        "name": "Entailment",
        "premise": "India became independent on 15 August 1947.",
        "hypothesis": "India gained independence in 1947.",
        "expected": "ENTAILMENT"
    },
    {
        "name": "Contradiction",
        "premise": "India became independent on 15 August 1947.",
        "hypothesis": "India gained independence in 1950.",
        "expected": "CONTRADICTION"
    },
    {
        "name": "Neutral",
        "premise": "India became independent on 15 August 1947.",
        "hypothesis": "India has a large population.",
        "expected": "NEUTRAL"
    }
]


for case in test_cases:

    result = predict_nli(
        premise=case["premise"],
        hypothesis=case["hypothesis"]
    )

    print("\nScenario:", case["name"])
    print("Expected:", case["expected"])
    print("Predicted:", result["nli_label"])

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

    print(
        "Result:",
        "PASS" if result["nli_label"] == case["expected"]
        else "REVIEW"
    )