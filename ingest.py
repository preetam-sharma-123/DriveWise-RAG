"""
ingest.py
---------
Step 1 of the Drive Wise pipeline: turn a raw brochure (PDF or .txt) into a
list of LangChain Document objects, each carrying rich metadata:
    - brand
    - model
    - section        (e.g. "Engine and Performance")
    - page            (best-effort page number)
    - doc_version

Sections are detected using "## Heading" style markers in the source text.
If you're feeding this a real PDF brochure that doesn't have clean section
markers, you'll want to either:
  (a) pre-process the PDF into a markdown-ish text file with "## Section"
      headers (recommended, keeps things simple and reliable), or
  (b) swap the SECTION_PATTERN / detection logic below for something that
      matches your brochure's actual heading style.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SECTION_PATTERN = re.compile(r"^##\s*(.+)$", re.MULTILINE)

# The canonical section vocabulary from the Drive Wise problem statement.
KNOWN_SECTIONS = [
    "Engine and Performance",
    "Mileage and Fuel Efficiency",
    "Safety",
    "Dimensions",
    "Interior and Comfort",
    "Infotainment and Connectivity",
]


@dataclass
class BrochureMeta:
    brand: str
    model: str
    doc_version: str = "v1.0"


def load_raw_text(path: str) -> str:
    """Load brochure text. Supports .txt/.md directly and .pdf via pypdf."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    return p.read_text(encoding="utf-8")


def split_into_sections(raw_text: str) -> List[dict]:
    """Split raw brochure text into {section, text} blocks based on ## headers.

    Falls back to a single "General" section if no headers are found.
    """
    matches = list(SECTION_PATTERN.finditer(raw_text))
    if not matches:
        return [{"section": "General", "text": raw_text.strip()}]

    blocks = []
    for i, m in enumerate(matches):
        section_name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        blocks.append({"section": section_name, "text": raw_text[start:end].strip()})
    return blocks


def chunk_brochure(
    path: str,
    meta: BrochureMeta,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Document]:
    """Full pipeline: load -> split into sections -> chunk each section ->
    attach metadata -> return LangChain Documents ready for embedding.
    """
    raw_text = load_raw_text(path)
    sections = split_into_sections(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    documents: List[Document] = []
    for block in sections:
        if not block["text"]:
            continue
        chunks = splitter.split_text(block["text"])
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "brand": meta.brand,
                        "model": meta.model,
                        "section": block["section"],
                        "doc_version": meta.doc_version,
                        "chunk_index": idx,
                        "source_file": Path(path).name,
                    },
                )
            )
    return documents


if __name__ == "__main__":
    # Quick smoke test using the sample brochure in data/
    meta = BrochureMeta(brand="Hyundai", model="Creta", doc_version="2026-v1.0")
    docs = chunk_brochure("data/hyundai_creta_2026_brochure.txt", meta)
    print(f"Produced {len(docs)} chunks\n")
    for d in docs[:3]:
        print("---")
        print("metadata:", d.metadata)
        print("content:", d.page_content[:120], "...")
