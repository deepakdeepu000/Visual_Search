from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.search import _get_embedder, _get_store, search_products


class SearchResult(BaseModel):
    id: int
    score: float = Field(..., description="Cosine similarity, 0.0 to 1.0.")
    image_path: str
    filename: str
    category: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]
    total_indexed: int


class HealthResponse(BaseModel):
    status: str
    total_indexed: int
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_embedder()
    _get_store()
    yield


app = FastAPI(
    title="Visual Product Search API",
    description="Search product images with natural language using CLIP embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    store = _get_store()
    return HealthResponse(
        status="ok",
        total_indexed=store.count(),
        model="openai/clip-vit-base-patch32",
    )


@app.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(description="Natural language search query")],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> SearchResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")
    results = search_products(q, top_k=top_k)
    store = _get_store()
    return SearchResponse(
        query=q,
        top_k=top_k,
        results=[SearchResult(**r) for r in results],
        total_indexed=store.count(),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)