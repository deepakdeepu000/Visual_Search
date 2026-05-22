from __future__ import annotations
from typing import Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "products"
VECTOR_DIM = 512
DISTANCE = Distance.COSINE
DB_PATH = "data/qdrant_db"


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(path=DB_PATH)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing_names = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE),
            )

    def upsert_batch(
        self,
        ids: list[int],
        vectors: list[np.ndarray],
        payloads: list[dict[str, Any]],
    ) -> None:
        points = [
            PointStruct(id=point_id, vector=vec.tolist(), payload=meta)
            for point_id, vec, meta in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
        ).points
        return [
            {"id": hit.id, "score": round(hit.score, 4), "payload": hit.payload}
            for hit in results
        ]

    def count(self) -> int:
        return self.client.count(COLLECTION_NAME).count
