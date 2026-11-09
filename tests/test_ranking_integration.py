
"""
Real-model integration test for Evidence Ranking.

Uses:
- Real Sentence Transformer model
- Real Wikipedia retrieval
- Real Tavily retrieval
- BM25 scoring
- Semantic similarity
- Combined relevance ranking
"""

from backend.data_contracts import Claim
from backend.evidence_retriever import retrieve_evidence
from backend.evidence_ranker import rank_evidence


def main():

    print("\n" + "=" * 65)
    print("VERI-CLAIM-AI: EVIDENCE RANKING INTEGRATION TEST")
    print("=" * 65)

    # ------------------------------------------
    # STEP 1: Create a sample claim
    # ------------------------------------------

    claim = Claim(
        claim_text="India gained independence in 1947.",
        original_text="India gained independence in 1947.",
        claim_type="historical"
    )

    print("\nCLAIM:")
    print(claim.claim_text)

    # ------------------------------------------
    # STEP 2: Retrieve real evidence
    # ------------------------------------------

    print("\nRetrieving evidence from Wikipedia and Tavily...")

    retrieval_report = retrieve_evidence([claim])

    evidence_list = retrieval_report.get("evidence", [])

    print("\nRETRIEVAL SUMMARY")
    print("-" * 65)

    print("Total evidence retrieved:", len(evidence_list))

    if not evidence_list:
        print("No evidence retrieved. Check provider status and errors.")
        print(retrieval_report.get("provider_status", {}))
        print(retrieval_report.get("errors", {}))
        return

    # ------------------------------------------
    # STEP 3: Rank evidence
    # ------------------------------------------

    print("\nRanking evidence using BM25 + Semantic Similarity...")

    ranked_results = rank_evidence(
        claim,
        evidence_list,
        top_k=5
    )

    # ------------------------------------------
    # STEP 4: Display ranking results
    # ------------------------------------------

    print("\nRANKED EVIDENCE")
    print("=" * 65)

    for result in ranked_results:

        evidence = result.evidence

        print(f"\nRANK: {result.rank}")

        print("Source:", evidence.source_title)

        print("Provider:", evidence.retrieval_provider)

        print("BM25 Score:", round(result.bm25_score, 4))

        print(
            "Semantic Score:",
            round(result.semantic_score, 4)
        )

        print(
            "Combined Score:",
            round(result.combined_score, 4)
        )

        print("Passage:", evidence.passage[:500])

        print("URL:", evidence.source_url)

        print("-" * 65)

    # ------------------------------------------
    # STEP 5: Final summary
    # ------------------------------------------

    print("\nINTEGRATION TEST SUMMARY")
    print("=" * 65)

    print("Claim:", claim.claim_text)

    print("Retrieved evidence:", len(evidence_list))

    print("Ranked evidence:", len(ranked_results))

    print("Embedding model: all-MiniLM-L6-v2")

    print("Ranking method: BM25 + Cosine Similarity")

    print("\nIntegration test completed.")


if __name__ == "__main__":
    main()
