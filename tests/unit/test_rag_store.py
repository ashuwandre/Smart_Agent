"""Unit tests for deterministic RAG indexing and retrieval behavior."""

from pathlib import Path

import faiss
import numpy as np

from customer_ops_agent.rag import store


def _fake_embeddings(texts: list[str]) -> np.ndarray:
    """Represent policy topics as fixed dimensions to avoid network calls in tests."""

    vectors = np.asarray(
        [
            [
                text.lower().count("refund") + 0.1,
                text.lower().count("billing") + 0.1,
                text.lower().count("router") + 0.1,
            ]
            for text in texts
        ],
        dtype=np.float32,
    )
    faiss.normalize_L2(vectors)
    return vectors


def test_build_index_and_search_return_three_scored_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The public workflow retains source metadata and ranks relevant policy first."""

    documents = {
        "refund_policy.md": "# Refund Policy\n\nEligible duplicate charges may be refunded.",
        "billing_policy.md": "# Billing Policy\n\nBilling occurs on the renewal date.",
        "technical_router.md": "# Router Guide\n\nRestart the router once.",
    }
    for filename, content in documents.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    monkeypatch.setattr(store, "_embed_texts", _fake_embeddings)

    assert store.build_index(tmp_path) == 3

    results = store.search("How can I get a refund?")

    assert len(results) == 3
    assert results[0]["metadata"] == {
        "filename": "refund_policy.md",
        "chunk_id": "refund_policy:0001",
        "title": "Refund Policy",
    }
    assert "Eligible duplicate charges" in results[0]["content"]
    assert all(isinstance(result["score"], float) for result in results)
    assert [result["score"] for result in results] == sorted(
        (result["score"] for result in results),
        reverse=True,
    )
