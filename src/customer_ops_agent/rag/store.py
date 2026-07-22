"""FAISS-backed retrieval over the markdown policy knowledge base."""

from __future__ import annotations

import re
from pathlib import Path
from threading import RLock
from typing import TypedDict

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# Paths are anchored to this module so callers can run the project from any
# working directory without silently indexing the wrong folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200

# Loading the project-local file is convenient for development, while
# override=False ensures deployment-provided environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ChunkMetadata(TypedDict):
    """Metadata retained alongside every FAISS vector."""

    filename: str
    chunk_id: str
    title: str


class ChunkRecord(TypedDict):
    """Stored source text and its retrieval metadata."""

    content: str
    metadata: ChunkMetadata


class SearchResult(ChunkRecord):
    """A retrieved chunk with cosine similarity."""

    score: float


# The index and records are swapped together after a successful build. This
# avoids exposing a partially built index if embedding generation fails.
_index: faiss.IndexFlatIP | None = None
_records: list[ChunkRecord] = []
_state_lock = RLock()


def _document_title(markdown: str, path: Path) -> str:
    """Read the first H1, falling back to a readable filename-derived title."""

    match = re.search(r"^#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("_", " ").title()


def _overlap_tail(text: str) -> str:
    """Keep a small word-aligned tail so facts spanning chunks remain searchable."""

    if len(text) <= CHUNK_OVERLAP:
        return text

    tail = text[-CHUNK_OVERLAP:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space >= 0 else tail


def _chunk_markdown(markdown: str, title: str) -> list[str]:
    """Chunk on markdown paragraph boundaries and retain document context."""

    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", markdown.strip())
        if block.strip()
    ]
    if not blocks:
        return []

    raw_chunks: list[str] = []
    current: list[str] = []

    for block in blocks:
        candidate = "\n\n".join([*current, block])
        if current and len(candidate) > CHUNK_SIZE:
            completed = "\n\n".join(current)
            raw_chunks.append(completed)
            overlap = _overlap_tail(completed)
            current = [part for part in (overlap, block) if part]
        else:
            current.append(block)

    if current:
        raw_chunks.append("\n\n".join(current))

    # Repeating the H1 on later chunks improves retrieval and gives every result
    # enough context to be cited independently.
    title_prefix = f"# {title}"
    return [
        chunk if chunk.startswith(title_prefix) else f"{title_prefix}\n\n{chunk}"
        for chunk in raw_chunks
    ]


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Generate normalized OpenAI embeddings suitable for cosine search."""

    if not texts:
        raise ValueError("At least one text is required for embedding.")

    client = OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = np.asarray([item.embedding for item in ordered], dtype=np.float32)

    if len(vectors) != len(texts):
        raise RuntimeError("Embedding provider returned an unexpected vector count.")
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise RuntimeError("Embedding provider returned invalid vectors.")

    # Inner product on unit vectors is cosine similarity, which gives callers
    # an intuitive score where larger values indicate a closer semantic match.
    faiss.normalize_L2(vectors)
    return vectors


def build_index(kb_dir: str | Path = DEFAULT_KB_DIR) -> int:
    """Read markdown files, embed their chunks, and build the FAISS index.

    Returns:
        The number of chunks indexed.
    """

    source_dir = Path(kb_dir)
    markdown_files = sorted(source_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No markdown knowledge-base files found in {source_dir}")

    records: list[ChunkRecord] = []
    for path in markdown_files:
        markdown = path.read_text(encoding="utf-8")
        title = _document_title(markdown, path)
        for position, content in enumerate(_chunk_markdown(markdown, title), start=1):
            records.append(
                {
                    "content": content,
                    "metadata": {
                        "filename": path.name,
                        "chunk_id": f"{path.stem}:{position:04d}",
                        "title": title,
                    },
                }
            )

    if not records:
        raise ValueError("Knowledge-base markdown files contained no indexable text.")

    vectors = _embed_texts([record["content"] for record in records])
    new_index = faiss.IndexFlatIP(vectors.shape[1])
    new_index.add(vectors)

    global _index, _records
    with _state_lock:
        _index = new_index
        _records = records

    return len(records)


def search(query: str) -> list[SearchResult]:
    """Return the three chunks most semantically similar to a query."""

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query must not be empty.")

    query_vector = _embed_texts([clean_query])

    with _state_lock:
        if _index is None:
            raise RuntimeError("RAG index is not built. Call build_index() first.")
        if query_vector.shape[1] != _index.d:
            raise RuntimeError("Query embedding dimension does not match the FAISS index.")

        result_count = min(3, len(_records))
        scores, positions = _index.search(query_vector, result_count)
        records_snapshot = list(_records)

    results: list[SearchResult] = []
    for score, position in zip(scores[0], positions[0], strict=True):
        # FAISS uses -1 when fewer neighbors exist; ignoring it makes the
        # function safe even for a very small custom knowledge base.
        if position < 0:
            continue
        record = records_snapshot[int(position)]
        results.append(
            {
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": float(score),
            }
        )

    return results
