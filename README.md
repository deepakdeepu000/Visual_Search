# Visual Product Search Engine

Most search engines match keywords — type "blue shoes" and they look for those exact words. This project does something different: it understands meaning. Type "something to wear on a rainy day" and it will surface boots, raincoats, and umbrellas — because it understands what you're looking for, not just what you typed.

This is a fully working semantic search engine built over a product image catalog. It understands both text and images in the same "language", so a written description and a product photo can be directly compared and matched.

---

## How It Works

The core idea is simple: convert every product image and every search query into a list of numbers that captures their meaning. Images and text that mean similar things end up with similar numbers. Search becomes a matter of finding the closest match.

Here's the full flow:

1. **Prepare the catalog** — product images are downloaded and standardised into a consistent format the model can process.
2. **Understand the images** — each image is passed through CLIP, an AI model trained on hundreds of millions of image-text pairs. It converts each image into a 512-number representation of its visual meaning.
3. **Store the meanings** — those 512-number vectors are saved into Qdrant, a database built specifically for this kind of similarity-based lookup. Think of it as a search index, but for meaning rather than keywords.
4. **Search by text** — when a user types a query, CLIP converts that text into the same 512-number format. Because CLIP was trained on both images and text together, the numbers are directly comparable.
5. **Find the closest match** — Qdrant finds whichever stored image vectors are mathematically closest to the query vector and returns them as results.
6. **Serve the results** — a lightweight API exposes this as a single search endpoint any frontend or service can call.

---

## Project Structure

```text
visual-search/
├── src/
│   ├── prepare_data.py   # Downloads and prepares product images
│   ├── embedder.py       # Converts images and text into vectors using CLIP
│   ├── store.py          # Manages the Qdrant vector database
│   ├── indexer.py        # Runs the full image → vector → database pipeline
│   ├── search.py         # CLI tool for testing queries directly
│   └── main.py           # FastAPI server exposing the search endpoint
└── data/
    ├── images/           # Prepared product images
    └── qdrant_db/        # Local vector database storage
```

---

## Requirements

- Python 3.11+
- `uv` package manager (handles dependencies cleanly without virtualenv setup)
- A GPU is optional — the model runs fine on CPU, just slower on large catalogs

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/deepakdeepu000/Visual_Search.git
cd visual-search
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Prepare the product catalog

```bash
uv run python src/prepare_data.py
```

Downloads 500 Fashion MNIST product images and standardises them for the model.

### 4. Index the catalog

```bash
uv run python src/indexer.py
```

Runs every image through CLIP and stores the resulting vectors in Qdrant. This only needs to be done once — or whenever the catalog changes.

### 5. Try a search from the command line

```bash
uv run python src/search.py "sneakers"
uv run python src/search.py "summer dress"
uv run python src/search.py "ankle boots"
```

### 6. Start the API server

```bash
uv run uvicorn src.main:app --reload --port 8000
```

### 7. Query the API

```bash
curl "http://127.0.0.1:8000/search?q=sneakers&top_k=5"
```

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

---

## API Reference

### `GET /health`
Returns whether the API is running, which model is loaded, and how many products are indexed.

### `GET /search`
| Parameter | Description |
|---|---|
| `q` | Your search query in plain English |
| `top_k` | How many results to return (1–20) |

---

## Notes

- The default CLIP model outputs 512-dimensional vectors. If you switch to `clip-vit-large-patch14`, update the vector dimension to `768` in `src/store.py` and reindex.
- To reset the database and start fresh, delete `data/qdrant_db/` and rerun the indexer.
- First load of CLIP takes a few seconds and uses ~350MB of RAM.

---

## What's Next

- Swap in your own product catalog instead of Fashion MNIST
- Add image-to-image search (find products that look like a photo)
- Move from local Qdrant storage to Qdrant Cloud for production use
- Add a frontend UI for browsing results visually
- Containerise with Docker for easy deployment
