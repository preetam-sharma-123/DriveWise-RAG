"""
rag_pipeline.py
----------------
Step 3 of the Drive Wise pipeline: given a brand, model, and user question,
retrieve the right brochure chunks and generate a grounded answer.

Implements the enhanced RAG design from the problem statement:
  5.1 Metadata filtering   -> filter by brand/model before ranking
  5.2 (chunking is handled upstream in ingest.py)
  5.3 Re-ranking            -> hybrid vector-similarity + keyword-overlap score
  5.4 Context window control -> only top-N chunks passed to the LLM
  6.  Source attribution    -> every answer lists brochure/section/page refs
  8.  Logging               -> every query is logged to logs/queries.jsonl
"""
from rank_bm25 import BM25Okapi
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from vectorstore import load_index

LOG_PATH = Path("logs/queries.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)
QUERY_CACHE = {}

RETRIEVE_K = 20          # candidates pulled from FAISS before re-ranking
CONTEXT_TOP_N = 4         # chunks actually sent to the LLM (context window control)

SYSTEM_PROMPT = (
    "You are Drive Wise, an assistant that answers questions about a specific "
    "car using ONLY the brochure excerpts provided below. If the excerpts do "
    "not contain the answer, say you don't have that information in the "
    "brochure rather than guessing. Keep answers concise and specific."
)


@dataclass
class RagAnswer:
    answer: str
    sources: List[dict] = field(default_factory=list)
    latency_seconds: float = 0.0


def _keyword_overlap_score(query: str, text: str) -> float:
    """Cheap lexical relevance signal used as one re-ranking factor.

    Counts how many distinct query terms (3+ chars) appear in the chunk,
    normalized by the number of query terms. This approximates what a
    cross-encoder re-ranker would do, without needing an extra model
    download or API call.
    """
    query_terms = {t for t in re.findall(r"[a-zA-Z]{3,}", query.lower())}
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for t in query_terms if t in text_lower)
    return hits / len(query_terms)

def _bm25_score(query: str, docs: List[Document]) -> List[float]:
    """
    Compute BM25 scores for a list of documents.
    """
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = query.lower().split()

    return bm25.get_scores(query_tokens)

def _rerank(query: str, scored_docs: List[tuple]) -> List[tuple]:
    """
    Hybrid reranking using:
    - FAISS similarity
    - BM25 lexical relevance
    - Keyword overlap
    - Section-aware boosting
    """
    if not scored_docs:
        return []

    # FAISS distance normalization
    distances = [d for _, d in scored_docs]
    max_d, min_d = max(distances), min(distances)
    spread = (max_d - min_d) or 1e-6

    docs_only = [doc for doc, _ in scored_docs]

    # BM25
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in docs_only
    ]

    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0

    reranked = []

    for idx, (doc, dist) in enumerate(scored_docs):

        # Vector similarity (0–1)
        vector_sim = 1 - (dist - min_d) / spread

        # BM25 similarity (0–1)
        bm25_sim = (
            bm25_scores[idx] / max_bm25
            if max_bm25 > 0
            else 0
        )

        # Keyword overlap
        keyword_sim = _keyword_overlap_score(
            query,
            doc.page_content
        )

        # Hybrid score
        combined = (
            0.50 * vector_sim +
            0.30 * bm25_sim +
            0.20 * keyword_sim
        )

        q = query.lower()
        section = doc.metadata.get("section", "")

        # Section-aware boosting
        if ("mileage" in q or "fuel" in q) and section == "Mileage and Fuel Efficiency":
            combined += 0.15

        elif ("airbag" in q or "safety" in q or "rating" in q) and section == "Safety":
            combined += 0.15

        elif ("dimension" in q or "size" in q) and section == "Dimensions":
            combined += 0.15

        elif ("engine" in q or "power" in q) and section == "Engine and Performance":
            combined += 0.15

        elif ("sunroof" in q or "comfort" in q) and section == "Interior and Comfort":
            combined += 0.15

        reranked.append((doc, combined))

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return reranked

def retrieve(brand: str, model: str, question: str) -> List[Document]:
    """Metadata-filtered retrieval + re-ranking + context window control.

    Brand/model matching is case-insensitive since users may type
    "hyundai creta" while the indexed metadata stores "Hyundai"/"Creta".
    """
    store = load_index()
    brand_norm = brand.strip().lower()
    model_norm = model.strip().lower()

    # FAISS's built-in filter does exact matching, so we pull a larger pool
    # of candidates unfiltered, then apply case-insensitive metadata
    # filtering ourselves before re-ranking.
    candidate_pool = store.similarity_search_with_score(
    question,
    k=RETRIEVE_K * 4
)
  
    scored = [
        (doc, dist)
        for doc, dist in candidate_pool
        if doc.metadata.get("brand", "").strip().lower() == brand_norm
        and doc.metadata.get("model", "").strip().lower() == model_norm
    ][:RETRIEVE_K]

    if not scored:
        return []

    reranked = _rerank(question, scored)
    top_docs = [doc for doc, _ in reranked[:CONTEXT_TOP_N]]
    return top_docs


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        blocks.append(
            f"[Excerpt {i} | Section: {d.metadata.get('section')} | "
            f"Source: {d.metadata.get('source_file')}]\n{d.page_content}"
        )
    return "\n\n".join(blocks)


def _build_sources(docs: List[Document]) -> List[dict]:
    sources = []
    for d in docs:
        sources.append(
            {
                "brochure": d.metadata.get("source_file"),
                "section": d.metadata.get("section"),
                "doc_version": d.metadata.get("doc_version"),
                "chunk_index": d.metadata.get("chunk_index"),
            }
        )
    return sources

def _extract_text(content) -> str:
    """response.content can be a plain string or a list of content blocks
    (e.g. [{'type': 'text', 'text': '...'}, ...]) depending on the model
    version. Normalize to a plain string either way.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)
def _log_event(event: dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def answer_query(brand: str, model: str, question: str) -> RagAnswer:
    start = time.time()

    cache_key = f"{brand.lower()}:{model.lower()}:{question.lower()}"

    # Cache lookup
    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]

    try:
        docs = retrieve(brand, model, question)

        # No retrieval results
        if not docs:
            latency = time.time() - start

            _log_event(
                {
                    "brand": brand,
                    "model": model,
                    "question": question,
                    "status": "no_results",
                    "latency_seconds": latency,
                }
            )

            result = RagAnswer(
                answer=(
                    f"I couldn't find brochure information for the "
                    f"{brand} {model} on this topic. "
                    "Try rephrasing, or check that a brochure for "
                    "this brand/model has been indexed."
                ),
                latency_seconds=latency,
            )

            QUERY_CACHE[cache_key] = result
            return result

        context = _format_context(docs)

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.2,
        )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Car: {brand} {model}\n\n"
            f"Brochure excerpts:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        response = None
        last_error = None

        for attempt in range(3):
            try:
                response = llm.invoke(prompt)
                break

            except Exception as exc:
                last_error = exc

                if attempt < 2:
                    time.sleep(5)

        if response is None:
            raise last_error

        answer_text = _extract_text(response.content)

        latency = time.time() - start
        sources = _build_sources(docs)

        _log_event(
            {
                "brand": brand,
                "model": model,
                "question": question,
                "status": "ok",
                "latency_seconds": latency,
                "sources": sources,
            }
        )

        result = RagAnswer(
            answer=answer_text,
            sources=sources,
            latency_seconds=latency,
        )

        QUERY_CACHE[cache_key] = result

        return result

    except Exception as exc:
        latency = time.time() - start

        _log_event(
            {
                "brand": brand,
                "model": model,
                "question": question,
                "status": "failed",
                "error": str(exc),
                "latency_seconds": latency,
            }
        )

        result = RagAnswer(
            answer=(
                "The brochure information was retrieved successfully, "
                "but answer generation failed because the Gemini API "
                "is temporarily unavailable or the usage quota has been exceeded."
            ),
            sources=_build_sources(docs) if "docs" in locals() else [],
            latency_seconds=latency,
        )

        QUERY_CACHE[cache_key] = result

        return result

        context = _format_context(docs)
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Car: {brand} {model}\n\n"
            f"Brochure excerpts:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )


        response = None
        last_error = None

        for attempt in range(3):
           try:
               response = llm.invoke(prompt)
               break
           except Exception as exc:
               last_error = exc

               if attempt < 2:
                   time.sleep(5)


        if response is None:
            raise last_error
        answer_text = _extract_text(response.content)

        latency = time.time() - start
        sources = _build_sources(docs)
        _log_event(
            {
                "brand": brand,
                "model": model,
                "question": question,
                "status": "ok",
                "latency_seconds": latency,
                "sources": sources,
            }
        )
        return RagAnswer(answer=answer_text, sources=sources, latency_seconds=latency)

    except Exception as exc:  # noqa: BLE001
        latency = time.time() - start
        _log_event(
            {
                "brand": brand,
                "model": model,
                "question": question,
                "status": "failed",
                "error": str(exc),
                "latency_seconds": latency,
            }
        )
        return RagAnswer(
            answer=(
                "The brochure information was retrieved successfully, "
                "but answer generation failed because the Gemini API "
                "is temporarily unavailable or the usage quota has been exceeded."
    ),
    sources=_build_sources(docs) if 'docs' in locals() else [],
    latency_seconds=latency,
)