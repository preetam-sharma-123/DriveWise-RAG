# Drive Wise — Prototype (Phase 1)

A minimal but *architecturally complete* implementation of the Drive Wise
problem statement, scoped to one brand/model (Hyundai Creta) so the pipeline
can be verified end-to-end before scaling to more brochures.

## What's implemented

| Problem statement requirement | File | Status |
|---|---|---|
| Structured chunking by brochure section | `ingest.py` | ✅ |
| Metadata (brand, model, section, version) | `ingest.py` | ✅ |
| Vector database (FAISS) | `vectorstore.py` | ✅ |
| Metadata filtering before retrieval | `rag_pipeline.py` (`retrieve`) | ✅ |
| Re-ranking of retrieved chunks | `rag_pipeline.py` (`_rerank`) | ✅ (hybrid vector + keyword score, see note below) |
| Context window control | `rag_pipeline.py` (`CONTEXT_TOP_N`) | ✅ |
| Source attribution | `rag_pipeline.py` (`_build_sources`) | ✅ |
| Query/latency/failure logging | `rag_pipeline.py` (`_log_event` → `logs/queries.jsonl`) | ✅ |
| Answer generation (Gemini) | `rag_pipeline.py` | ✅ |
| Page number in metadata | — | ⏳ not yet — see "Next steps" |
| Evaluation metrics (correctness/faithfulness/context relevance) | — | ⏳ not yet |

**Re-ranking note:** a true cross-encoder re-ranker (e.g. from
`sentence-transformers`) would need a model download from huggingface.co,
which isn't reachable from this sandbox's network. Instead, `_rerank()` uses
a hybrid score combining FAISS vector similarity with a lightweight keyword
overlap signal. This is a placeholder — swap it for a real cross-encoder
once you're running this locally or in Colab with full internet access.

## Setup

```bash
cd drivewise
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"   # https://aistudio.google.com/apikey

# Build the FAISS index from brochures in data/
python vectorstore.py --build

# Ask questions
python app.py
```

## Project layout

```
drivewise/
├── data/
│   └── hyundai_creta_2026_brochure.txt   # sample brochure (replace with real PDFs)
├── ingest.py          # load + section-split + chunk + metadata
├── vectorstore.py      # embed + FAISS index build/load
├── rag_pipeline.py      # retrieve -> filter -> re-rank -> generate -> log
├── app.py              # CLI
├── requirements.txt
└── logs/
    └── queries.jsonl    # created after first run
```

## Adding a real brochure

1. Download the brochure PDF (e.g. from the manufacturer's site).
2. Either:
   - Drop the PDF straight into `data/` — `ingest.py` will extract text via
     `pypdf`, but it won't have clean `## Section` headers, so everything
     will land in one "General" section. Fine for a first test, not ideal
     for metadata-based filtering by section.
   - **Recommended:** convert/clean it into a `.txt`/`.md` file with
     `## Engine and Performance`, `## Safety`, etc. headers matching the six
     sections in the problem statement. This gets you the full benefit of
     section-aware metadata filtering.
3. Add an entry to `BROCHURE_REGISTRY` in `vectorstore.py` mapping the
   filename to a `BrochureMeta(brand=..., model=..., doc_version=...)`.
4. Re-run `python vectorstore.py --build`.

## Next steps (not yet built)

- **Page numbers**: `pypdf` can give per-page text; track page index during
  chunking so `[Excerpt N | Page: X]` shows in source attribution.
- **Evaluation harness**: a small script that runs a fixed question set and
  scores Answer Correctness / Faithfulness / Context Relevance (e.g. using
  Gemini itself as a judge, or a RAGAS-style library).
- **Web interface**: wrap `answer_query()` in a Streamlit app (you already
  have Streamlit experience from Netra AI) with brand/model dropdowns.
- **Multi-brand scale-out**: once Creta works end-to-end, add more
  brand/model brochures to `BROCHURE_REGISTRY`.
