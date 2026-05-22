from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import CLIPEmbedder
from src.store import VectorStore

_embedder: CLIPEmbedder | None = None
_store: VectorStore | None = None


def _get_embedder() -> CLIPEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = CLIPEmbedder()
    return _embedder


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def search_products(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    embedder = _get_embedder()
    store = _get_store()
    query_vector = embedder.embed_text(query)
    raw_results = store.search(query_vector, top_k=top_k)

    return [
        {
            "id": r["id"],
            "score": r["score"],
            "image_path": r["payload"].get("image_path", ""),
            "filename": r["payload"].get("filename", ""),
            "category": r["payload"].get("category", "unknown"),
        }
        for r in raw_results
    ]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "sneakers"
    results = search_products(query, top_k=5)
    for rank, r in enumerate(results, start=1):
        print(
            f"#{rank} score={r['score']:.4f} "
            f"category={r['category']:<12} file={r['filename']}"
        )
