"""
eval_harness.py
----------------
Step 7 of the Drive Wise pipeline: evaluation and quality tracking.

Runs a fixed set of test questions (with known reference facts) through the
RAG pipeline and scores three dimensions, as specified in the problem
statement:
  - Answer Correctness: does the answer match the brochure's actual facts?
  - Faithfulness: is the answer strictly grounded in the retrieved context
    (no hallucinated claims not present in the retrieved chunks)?
  - Context Relevance: are the retrieved chunks actually relevant to the
    question?

Uses Gemini itself as an LLM judge, since building a labeled ground-truth
dataset by hand doesn't scale, and cross-encoder/embedding-based judges would
need extra model downloads. This is a common, well-documented pattern (e.g.
RAGAS uses the same "LLM-as-judge" approach) — treat these scores as a
useful signal, not an infallible ground truth.

Usage:
    python eval_harness.py
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_pipeline import retrieve, answer_query, _extract_text

RESULTS_PATH = Path("logs/eval_results.jsonl")
RESULTS_PATH.parent.mkdir(exist_ok=True)

JUDGE_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Test set: each item has a brand/model/question plus the key facts the
# answer SHOULD contain, drawn from the actual brochure content. Extend this
# as you add more brochures or want deeper coverage per section.
# ---------------------------------------------------------------------------
TEST_SET = [
    {
        "brand": "Hyundai",
        "model": "Creta",
        "question": "What's the ARAI-certified mileage of the petrol manual variant?",
        "reference_facts": "Approximately 17.4 km/l for the petrol manual variant.",
    },
    {
        "brand": "Hyundai",
        "model": "Creta",
        "question": "How many airbags does it have as standard?",
        "reference_facts": "Six airbags (driver, passenger, side, and curtain) standard across all variants.",
    },
    {
        "brand": "Hyundai",
        "model": "Creta",
        "question": "What are the overall dimensions?",
        "reference_facts": "4,300 mm length, 1,790 mm width, 1,635 mm height, 2,610 mm wheelbase.",
    },
    {
        "brand": "Tata",
        "model": "Nexon",
        "question": "What safety rating has it received?",
        "reference_facts": "5-star Bharat NCAP rating for adult and child occupant protection.",
    },
    {
        "brand": "Tata",
        "model": "Nexon",
        "question": "What engine options are available?",
        "reference_facts": "1.2L 3-cylinder turbo petrol (up to 120 PS/170 Nm) and 1.5L diesel (up to 115 PS/260 Nm); CNG also available.",
    },
    {
        "brand": "Tata",
        "model": "Nexon",
        "question": "Does it have a sunroof?",
        "reference_facts": "Yes, a panoramic sunroof operable via voice command.",
    },
]


@dataclass
class EvalResult:
    question: str
    brand: str
    model: str
    answer: str
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_correctness: float = 0.0
    judge_notes: str = ""
    latency_seconds: float = 0.0


JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge for a RAG (Retrieval-Augmented
Generation) system. Score the following on three dimensions, each from 0.0 to 1.0.

Question: {question}

Retrieved context given to the model:
{context}

Reference facts (ground truth from the brochure):
{reference_facts}

Generated answer:
{answer}

Score these three dimensions:
1. context_relevance: How relevant is the retrieved context to answering the question?
   (1.0 = highly relevant, 0.0 = irrelevant)
2. faithfulness: Is the generated answer strictly grounded in the retrieved context,
   with no claims that aren't supported by it? (1.0 = fully grounded, 0.0 = hallucinated)
3. answer_correctness: Does the generated answer match the reference facts?
   (1.0 = fully correct and complete, 0.0 = wrong or missing the key facts)

Respond with ONLY a JSON object, no other text, in this exact format:
{{"context_relevance": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0, "notes": "brief explanation"}}
"""


def _judge(question: str, context: str, reference_facts: str, answer: str) -> dict:
    llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0.0)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        reference_facts=reference_facts,
        answer=answer
    )

    try:
        response = llm.invoke(prompt)

        raw = _extract_text(response.content).strip()

        # Strip markdown code fences if the judge wraps its JSON in them
        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception as e:
        return {
            "context_relevance": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "notes": f"Judge failed: {e}"
        }

def run_eval() -> List[EvalResult]:
    results = []
    for case in TEST_SET:
        start = time.time()
        brand, model, question = case["brand"], case["model"], case["question"]

        docs = retrieve(brand, model, question)
        context = "\n\n".join(d.page_content for d in docs)

        rag_answer = answer_query(brand, model, question)

        judged = _judge(question, context, case["reference_facts"], rag_answer.answer)
        latency = time.time() - start

        result = EvalResult(
            question=question,
            brand=brand,
            model=model,
            answer=rag_answer.answer,
            context_relevance=judged.get("context_relevance", 0.0),
            faithfulness=judged.get("faithfulness", 0.0),
            answer_correctness=judged.get("answer_correctness", 0.0),
            judge_notes=judged.get("notes", ""),
            latency_seconds=latency,
        )
        results.append(result)

        with RESULTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.__dict__) + "\n")

        print(f"[{brand} {model}] {question}")
        print(f"  Context Relevance:  {result.context_relevance:.2f}")
        print(f"  Faithfulness:       {result.faithfulness:.2f}")
        print(f"  Answer Correctness: {result.answer_correctness:.2f}")
        print(f"  Notes: {result.judge_notes}")
        print()

    return results


def print_summary(results: List[EvalResult]) -> None:
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    avg_cr = sum(r.context_relevance for r in results) / n
    avg_f = sum(r.faithfulness for r in results) / n
    avg_ac = sum(r.answer_correctness for r in results) / n
    avg_latency = sum(r.latency_seconds for r in results) / n

    print("=" * 50)
    print(f"SUMMARY ({n} test cases)")
    print("=" * 50)
    print(f"Avg Context Relevance:  {avg_cr:.2f}")
    print(f"Avg Faithfulness:       {avg_f:.2f}")
    print(f"Avg Answer Correctness: {avg_ac:.2f}")
    print(f"Avg Latency:            {avg_latency:.2f}s")

    weak_spots = [r for r in results if min(r.context_relevance, r.faithfulness, r.answer_correctness) < 0.6]
    if weak_spots:
        print(f"\n{len(weak_spots)} weak spot(s) worth reviewing:")
        for r in weak_spots:
            print(f"  - [{r.brand} {r.model}] \"{r.question}\"")


if __name__ == "__main__":
    print(f"Running evaluation on {len(TEST_SET)} test cases...\n")
    results = run_eval()
    print_summary(results)
    print(f"\nFull results saved to {RESULTS_PATH}")