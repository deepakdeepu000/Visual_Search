from pathlib import Path
from datasets import load_dataset
from PIL import Image

IMAGES_DIR = Path("data/images")
NUM_IMAGES = 500
SEED = 42

LABEL_NAMES = [
    "tshirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]


def prepare_dataset() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("fashion_mnist", split="train")
    dataset = dataset.shuffle(seed=SEED).select(range(NUM_IMAGES))

    for idx, item in enumerate(dataset):
        label_name = LABEL_NAMES[item["label"]]
        filename = f"{idx:04d}_{label_name}.png"
        filepath = IMAGES_DIR / filename
        img: Image.Image = item["image"]
        img = img.resize((224, 224), Image.LANCZOS)
        img.convert("RGB").save(filepath)


if __name__ == "__main__":
    prepare_dataset()
