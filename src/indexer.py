from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import CLIPEmbedder
from src.store import VectorStore

IMAGES_DIR = Path("data/images")
BATCH_SIZE = 32


def extract_category(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_", maxsplit=1)
    return parts[1] if len(parts) > 1 else "unknown"


def index_images() -> None:
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}/")
        print("Run: uv run python src/prepare_data.py")
        return

    embedder = CLIPEmbedder()
    store = VectorStore()

    if store.count() >= len(image_paths):
        print(f"Already indexed {store.count()} images.")
        print("Delete data/qdrant_db/ to force a re-index.")
        return

    batch_ids: list[int] = []
    batch_vectors = []
    batch_payloads: list[dict] = []
    total_indexed = 0

    for idx, image_path in enumerate(image_paths):
        try:
            vector = embedder.embed_image(image_path)
        except Exception as e:
            print(f"Skipping {image_path.name}: {e}")
            continue

        payload = {
            "image_path": str(image_path),
            "filename": image_path.name,
            "category": extract_category(image_path.name),
        }

        batch_ids.append(idx)
        batch_vectors.append(vector)
        batch_payloads.append(payload)

        if len(batch_ids) == BATCH_SIZE:
            store.upsert_batch(batch_ids, batch_vectors, batch_payloads)
            total_indexed += len(batch_ids)
            print(f"Indexed {total_indexed}/{len(image_paths)} images...")
            batch_ids.clear()
            batch_vectors.clear()
            batch_payloads.clear()

    if batch_ids:
        store.upsert_batch(batch_ids, batch_vectors, batch_payloads)
        total_indexed += len(batch_ids)

    print(f"Indexing complete: {total_indexed} images stored.")


if __name__ == "__main__":
    index_images()
