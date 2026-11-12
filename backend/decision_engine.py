"""
Decision Engine
---------------

Makes evidence-based claim classifications using:
- Evidence relevance scores
- NLI verification results
- Source reliability assessments
- Independent source corroboration

Possible classifications:
    SUPPORTED
    CONTRADICTED
    UNVERIFIABLE

Important:
    NLI confidence is not factual certainty.
    Source reliability is a heuristic, not proof of truth.
    Absence of evidence is not contradiction.
"""

from collections import defaultdict
from statistics import mean
from urllib.parse import urlparse

from backend.data_contracts import (
    Claim,
    EvidenceVerification,
    SourceReliability,
    VerificationResult,
)

from backend.evidence_ranker import RankedEvidence


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_RELEVANCE_SCORE = 0.40
MIN_NLI_CONFIDENCE = 0.85
MIN_RELIABILITY_SCORE = 0.60

MIN_INDEPENDENT_SOURCES = 2

DECISIVE_NLI_LABELS = {
    "ENTAILMENT",
    "CONTRADICTION",
}


# --------------------------------------------------
# SOURCE INDEPENDENCE
# --------------------------------------------------

# Common multi-label public suffixes.
# This is a lightweight approximation, not a full
# Public Suffix List implementation.

MULTI_LABEL_SUFFIXES = {
    "co.uk",
    "org.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.in",
    "com.in",
    "org.in",
    "net.in",
}


def get_source_domain(url):
    """
    Extract a normalized registrable-domain approximation.

    Examples:
        https://www.example.com/article
        -> example.com

        https://news.example.com/article
        -> example.com

    This helps avoid counting subdomains of the same
    organization as independent sources.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("Source URL must be a non-empty string.")

    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Invalid source URL.")

    hostname = parsed.hostname.lower().strip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Handle IP addresses and localhost-like hosts.
    if hostname.replace(".", "").isdigit() or "." not in hostname:
        return hostname

    parts = hostname.split(".")

    if len(parts) < 2:
        return hostname

    last_two = ".".join(parts[-2:])

    if last_two in MULTI_LABEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])

    return last_two


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_decision_inputs(
    claim,
    ranked_evidence,
    nli_results,
    reliability_results,
):
    """Validate the input objects and their identifiers."""

    if not isinstance(claim, Claim):
        raise TypeError("claim must be a Claim object.")

    if not isinstance(ranked_evidence, list):
        raise TypeError("ranked_evidence must be a list.")

    if not isinstance(nli_results, list):
        raise TypeError("nli_results must be a list.")

    if not isinstance(reliability_results, list):
        raise TypeError("reliability_results must be a list.")

    evidence_ids = set()

    for item in ranked_evidence:

        if not isinstance(item, RankedEvidence):
            raise TypeError(
                "Every ranked evidence item must be RankedEvidence."
            )

        if item.evidence.claim_id != claim.claim_id:
            raise ValueError(
                "Ranked evidence belongs to a different claim."
            )

        if item.evidence.evidence_id in evidence_ids:
            raise ValueError("Duplicate evidence ID in ranked evidence.")

        if not isinstance(item.combined_score, (int, float)):
            raise TypeError("Combined relevance score must be numeric.")

        evidence_ids.add(item.evidence.evidence_id)

    nli_ids = set()

    for result in nli_results:

        if not isinstance(result, EvidenceVerification):
            raise TypeError(
                "Every NLI result must be EvidenceVerification."
            )

        if result.claim_id != claim.claim_id:
            raise ValueError(
                "NLI result belongs to a different claim."
            )

        if result.evidence_id not in evidence_ids:
            raise ValueError(
                "NLI result references unknown evidence."
            )

        if result.evidence_id in nli_ids:
            raise ValueError("Duplicate NLI result for an evidence ID.")

        nli_ids.add(result.evidence_id)

    reliability_ids = set()

    for result in reliability_results:

        if not isinstance(result, SourceReliability):
            raise TypeError(
                "Every reliability result must be SourceReliability."
            )

        if result.evidence_id not in evidence_ids:
            raise ValueError(
                "Reliability result references unknown evidence."
            )

        if result.evidence_id in reliability_ids:
            raise ValueError(
                "Duplicate reliability result for an evidence ID."
            )

        reliability_ids.add(result.evidence_id)


# --------------------------------------------------
# EVIDENCE PREPARATION
# --------------------------------------------------

def prepare_evidence(
    ranked_evidence,
    nli_results,
    reliability_results,
):
    """
    Match evidence with its NLI and reliability assessments.

    Evidence without either assessment is excluded.
    """

    nli_map = {
        result.evidence_id: result
        for result in nli_results
    }

    reliability_map = {
        result.evidence_id: result
        for result in reliability_results
    }

    prepared = []

    for ranked_item in ranked_evidence:

        evidence = ranked_item.evidence
        evidence_id = evidence.evidence_id

        nli = nli_map.get(evidence_id)
        reliability = reliability_map.get(evidence_id)

        # Do not guess missing assessments.
        if nli is None or reliability is None:
            continue

        if ranked_item.combined_score < MIN_RELEVANCE_SCORE:
            continue

        if reliability.reliability_score < MIN_RELIABILITY_SCORE:
            continue

        prepared.append({
            "evidence": evidence,
            "ranked": ranked_item,
            "nli": nli,
            "reliability": reliability,
        })

    return prepared


# --------------------------------------------------
# DECISIVE EVIDENCE
# --------------------------------------------------

def identify_decisive_evidence(prepared_evidence):
    """
    Keep evidence with sufficiently confident
    entailment or contradiction predictions.
    """

    decisive = []

    for item in prepared_evidence:

        nli = item["nli"]

        if nli.nli_label not in DECISIVE_NLI_LABELS:
            continue

        if nli.nli_confidence < MIN_NLI_CONFIDENCE:
            continue

        decisive.append(item)

    return decisive


# --------------------------------------------------
# INDEPENDENT SOURCE GROUPING
# --------------------------------------------------

def group_independent_sources(decisive_evidence):
    """
    Group evidence by normalized source domain.

    If one domain contains both entailment and
    contradiction evidence, that domain is considered
    internally conflicting and excluded from both sides.

    Only one source contribution per domain is counted.
    """

    domain_labels = defaultdict(lambda: defaultdict(list))

    for item in decisive_evidence:

        domain = get_source_domain(
            item["evidence"].source_url
        )

        label = item["nli"].nli_label

        domain_labels[domain][label].append(item)

    supported_sources = []
    contradicted_sources = []
    internally_conflicting_domains = []

    for domain, labels in domain_labels.items():

        has_support = bool(labels.get("ENTAILMENT"))
        has_contradiction = bool(labels.get("CONTRADICTION"))

        if has_support and has_contradiction:
            internally_conflicting_domains.append(domain)
            continue

        if has_support:
            # Select the strongest evidence from this domain.
            best = max(
                labels["ENTAILMENT"],
                key=lambda item: item["nli"].nli_confidence,
            )

            supported_sources.append(best)

        elif has_contradiction:
            best = max(
                labels["CONTRADICTION"],
                key=lambda item: item["nli"].nli_confidence,
            )

            contradicted_sources.append(best)

    return (
        supported_sources,
        contradicted_sources,
        internally_conflicting_domains,
    )


# --------------------------------------------------
# AGGREGATE NLI RESULT
# --------------------------------------------------

def calculate_aggregate_nli(prepared_evidence):
    """
    Calculate an aggregate NLI tendency from eligible evidence.

    This is a summary of model predictions, NOT the final
    factual classification.
    """

    if not prepared_evidence:
        return "NEUTRAL", 0.0

    labels = [
        "ENTAILMENT",
        "CONTRADICTION",
        "NEUTRAL",
    ]

    average_scores = {
        "ENTAILMENT": mean(
            item["nli"].entailment_score
            for item in prepared_evidence
        ),
        "CONTRADICTION": mean(
            item["nli"].contradiction_score
            for item in prepared_evidence
        ),
        "NEUTRAL": mean(
            item["nli"].neutral_score
            for item in prepared_evidence
        ),
    }

    aggregate_label = max(
        labels,
        key=lambda label: average_scores[label],
    )

    aggregate_confidence = mean(
        item["nli"].nli_confidence
        for item in prepared_evidence
        if item["nli"].nli_label == aggregate_label
    ) if any(
        item["nli"].nli_label == aggregate_label
        for item in prepared_evidence
    ) else average_scores[aggregate_label]

    return aggregate_label, round(aggregate_confidence, 4)


# --------------------------------------------------
# EXPLANATION GENERATION
# --------------------------------------------------

def generate_decision_explanation(
    classification,
    prepared_evidence,
    supported_sources,
    contradicted_sources,
    conflicting_domains,
):
    """Generate a human-readable decision explanation."""

    relevant_count = len(prepared_evidence)
    support_count = len(supported_sources)
    contradiction_count = len(contradicted_sources)

    if classification == "SUPPORTED":

        return (
            f"The claim is classified as SUPPORTED because "
            f"{support_count} independent sources provide "
            f"high-confidence supporting evidence. "
            f"{relevant_count} evidence items passed the "
            f"relevance and reliability filters. "
            f"No credible contradictory evidence was identified."
        )

    if classification == "CONTRADICTED":

        return (
            f"The claim is classified as CONTRADICTED because "
            f"{contradiction_count} independent sources provide "
            f"high-confidence contradictory evidence. "
            f"{relevant_count} evidence items passed the "
            f"relevance and reliability filters. "
            f"No credible supporting evidence was identified."
        )

    if not prepared_evidence:

        return (
            "The claim is classified as UNVERIFIABLE because "
            "no evidence passed the relevance and source "
            "reliability filters, or required assessments were "
            "unavailable."
        )

    if conflicting_domains:

        return (
            "The claim is classified as UNVERIFIABLE because "
            "supporting and contradictory predictions were found "
            "within the same source domain(s): "
            f"{', '.join(conflicting_domains)}. "
            "These sources were not counted as independent "
            "corroboration."
        )

    if support_count and contradiction_count:

        return (
            "The claim is classified as UNVERIFIABLE because "
            f"{support_count} independent source(s) support it "
            f"while {contradiction_count} independent source(s) "
            "contradict it. The evidence is conflicting."
        )

    return (
        "The claim is classified as UNVERIFIABLE because the "
        "available evidence does not provide sufficient "
        "independent, high-confidence corroboration. "
        f"Eligible evidence items: {relevant_count}; "
        f"supporting independent sources: {support_count}; "
        f"contradictory independent sources: {contradiction_count}."
    )


# --------------------------------------------------
# MAIN DECISION FUNCTION
# --------------------------------------------------

def decide_claim(
    claim,
    ranked_evidence,
    nli_results,
    reliability_results,
):
    """
    Make a final evidence-based decision for one claim.

    Returns:
        VerificationResult
    """

    validate_decision_inputs(
        claim,
        ranked_evidence,
        nli_results,
        reliability_results,
    )

    prepared = prepare_evidence(
        ranked_evidence,
        nli_results,
        reliability_results,
    )

    decisive = identify_decisive_evidence(prepared)

    (
        supported_sources,
        contradicted_sources,
        conflicting_domains,
    ) = group_independent_sources(decisive)

    support_count = len(supported_sources)
    contradiction_count = len(contradicted_sources)

    # Strict decision gates.
    if (
        support_count >= MIN_INDEPENDENT_SOURCES
        and contradiction_count == 0
        and not conflicting_domains
    ):
        classification = "SUPPORTED"

    elif (
        contradiction_count >= MIN_INDEPENDENT_SOURCES
        and support_count == 0
        and not conflicting_domains
    ):
        classification = "CONTRADICTED"

    else:
        classification = "UNVERIFIABLE"

    aggregate_label, aggregate_confidence = (
        calculate_aggregate_nli(prepared)
    )

    explanation = generate_decision_explanation(
        classification,
        prepared,
        supported_sources,
        contradicted_sources,
        conflicting_domains,
    )

    # Include all evidence that passed relevance and reliability
    # filters, including evidence involved in conflicts.
    considered_evidence_ids = [
        item["evidence"].evidence_id
        for item in prepared
    ]

    return VerificationResult(
        claim_id=claim.claim_id,
        classification=classification,
        nli_label=aggregate_label,
        nli_confidence=aggregate_confidence,
        evidence_ids=considered_evidence_ids,
        explanation=explanation,
    )