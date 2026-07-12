"""
app.py
------
Minimal CLI for Drive Wise. Run this after building the FAISS index.

Usage:
    export GOOGLE_API_KEY="your-key-here"
    python vectorstore.py --build   # once, or whenever brochures change
    python app.py
"""

from rag_pipeline import answer_query
from vectorstore import BROCHURE_REGISTRY


def main():
    print("=" * 60)
    print("Drive Wise — Brochure-Grounded Car Assistant")
    print("=" * 60)

    available = sorted({(m.brand, m.model) for m in BROCHURE_REGISTRY.values()})
    print("Available brand/model combinations:")
    for brand, model in available:
        print(f"  - {brand} {model}")
    print()

    brand = input("Enter car brand: ").strip()
    model = input("Enter car model: ").strip()

    print(f"\nAsk anything about the {brand} {model}. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        result = answer_query(brand, model, question)
        print(f"\nDrive Wise: {result.answer}\n")
        if result.sources:
            print("Sources:")
            for s in result.sources:
                print(
                    f"  - {s['brochure']} | Section: {s['section']} "
                    f"| Version: {s['doc_version']}"
                )
        print(f"(answered in {result.latency_seconds:.2f}s)\n")


if __name__ == "__main__":
    main()
