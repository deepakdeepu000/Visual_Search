from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPEmbedder:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
        if hasattr(raw, "pooler_output"):
            features = raw.pooler_output
        elif hasattr(raw, "image_embeds"):
            features = raw.image_embeds
        else:
            features = raw
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()

    def embed_text(self, query: str) -> np.ndarray:
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            raw = self.model.get_text_features(**inputs)
        if hasattr(raw, "pooler_output"):
            features = raw.pooler_output
        elif hasattr(raw, "text_embeds"):
            features = raw.text_embeds
        else:
            features = raw
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()
