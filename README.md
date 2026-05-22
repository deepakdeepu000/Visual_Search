# Visual Product Search Engine with CLIP + Qdrant

A complete starter project for building a text-to-image product search engine using:
- **CLIP** for image and text embeddings
- **Qdrant** as the vector database
- **FastAPI** for the API layer
- **Fashion MNIST** as the demo product image dataset

This project is based on the architecture described in the attached PDF and implements the full flow: dataset preparation, embedding generation, vector indexing, semantic search, and API serving.

## Project Structure

```text
visual-search/
├── pyproject.toml
├── README.md
├── data/
│   ├── images/
│   └── qdrant_db/
└── src/
    ├── prepare_data.py
    ├── embedder.py
    ├── store.py
    ├── indexer.py
    ├── search.py
    └── main.py
```

## Features

- Downloads and prepares 500 Fashion MNIST images
- Converts images into CLIP embeddings
- Stores vectors in a persistent local Qdrant database
- Supports natural-language text search over images
- Exposes `/health` and `/search` FastAPI endpoints
- Uses cosine similarity over normalized CLIP embeddings

## Requirements

- Python 3.11+
- `uv` package manager
- Optional NVIDIA GPU for faster CLIP inference

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

### 1. Create and enter the project

```bash
git clone https://github.com/deepakdeepu000/Visual_Search.git
cd visual-search
```

### 2. Install dependencies

```bash
uv sync
```

If you want CPU-only PyTorch, you can simplify `pyproject.toml` and remove the CUDA index configuration.

## Run the project

### Step 1: Prepare dataset

```bash
uv run python src/prepare_data.py
```

This downloads Fashion MNIST, selects 500 shuffled images, resizes them to 224x224, converts them to RGB, and saves them to `data/images/`.

### Step 2: Build the vector index

```bash
uv run python src/indexer.py
```

This loads CLIP, generates embeddings for all saved images, and stores them in Qdrant under the `products` collection.

### Step 3: Test text search from CLI

```bash
uv run python src/search.py "sneakers"
uv run python src/search.py "summer dress"
uv run python src/search.py "ankle boots"
```

### Step 4: Start the API server

```bash
uv run uvicorn src.main:app --reload --port 8000
```

### Step 5: Test the API

Health endpoint:

```bash
curl "http://127.0.0.1:8000/health"
```

Search endpoint:

```bash
curl "http://127.0.0.1:8000/search?q=sneakers&top_k=5"
```

## API Endpoints

### `GET /health`
Returns API status, model name, and total indexed items.

### `GET /search`
Query parameters:
- `q`: natural language search query
- `top_k`: number of results to return, from 1 to 20

Example response:

```json
{
  "query": "sneakers",
  "top_k": 5,
  "results": [
    {
      "id": 3,
      "score": 0.2831,
      "image_path": "data/images/0003_sneaker.png",
      "filename": "0003_sneaker.png",
      "category": "sneaker"
    }
  ],
  "total_indexed": 500
}
```

## How it works

1. **Prepare data**: Download Fashion MNIST images and store them locally.
2. **Embed images**: Use CLIP image encoder to generate 512-dimensional vectors.
3. **Store vectors**: Save vectors plus metadata in Qdrant.
4. **Embed text**: Convert a user query into the same vector space.
5. **Search**: Use cosine similarity in Qdrant to retrieve nearest image vectors.
6. **Serve results**: Return ranked matches through CLI or API.

## Notes

- The default model is `openai/clip-vit-base-patch32`, which outputs 512-dimensional embeddings.
- If you switch to `openai/clip-vit-large-patch14`, update the Qdrant vector dimension from `512` to `768` in `src/store.py` and rebuild the index.
- To reset the database, delete `data/qdrant_db/` and rerun indexing.
- The first CLIP load can take a few seconds and ~350MB RAM.


## Future improvements

- Upload your own product catalog instead of Fashion MNIST
- Add image-to-image search
- Use Qdrant Cloud instead of local storage
- Add a frontend UI for browsing results
- Add Docker support
- Add batch indexing progress tracking
