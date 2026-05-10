"""
JARVIS — World Model
====================
Mission: §23.3 — CLIP encoder for object visual embeddings + text-query
         matching. Used by:
            • ObservationBuilder._build_object_obs (every YOLO /
              OWLv2 object detection gets embedded for dedup +
              cost-function similarity).
            • WorldQueryTools.find_object — text query → embedding,
              cosine over the registered object embeddings → "where
              are my keys?" gets a structured answer.

         Lazy-import strategy: open_clip is a heavy dep (~150MB
         weights on first run + torch tensor ops on every encode).
         Construction is wrapped so a missing install logs a warning
         and falls back to NullCLIPEncoder; the rest of the system
         still works — object dedup just degrades to "every detection
         is a new entity" until the dep lands.

         Singleton — instantiate once in the orchestrator and pass to
         ObservationBuilder + the world-model query layer.

Modules: modules/vision/clip_encoder.py
Classes: CLIPEncoder, NullCLIPEncoder
Spec:    new 2.md §23.3.

#todo: ViT-L/14 fallback when ViT-B/32 produces poor recall on
       visually-similar objects (multiple wallets, multiple cups).
       Tunable via config.world_model.clip_encoder.model_name.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


class CLIPEncoder:
    """OpenCLIP wrapper. ViT-B/32 chosen as default — good balance of
    speed (~3ms image embed on RTX 4070 Ti) and quality. Bump to
    ViT-L/14 if recall on visually-similar objects is poor.

    Construction can fail (missing open_clip, weights download
    timeout, no CUDA). Callers that want a guaranteed encoder
    should use `try_load()` which returns CLIPEncoder OR
    NullCLIPEncoder."""

    DEFAULT_MODEL = "ViT-B-32"
    DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
    DEFAULT_DIM = 512

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str = "cuda",
    ) -> None:
        # Imports happen here, not at module load, so a missing
        # open_clip doesn't crash anything that just wants the symbol.
        import torch
        import open_clip  # type: ignore[import-not-found]

        self.device = device if torch.cuda.is_available() else "cpu"
        self._torch = torch
        # open_clip returns (model, train_transform, val_transform); we
        # only want the val transform. Pylance can't see the tuple
        # types accurately so we annotate locally.
        self.model, _, self.preprocess = (  # type: ignore[misc]
            open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device,
            )
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        # Probe the embedding dimension by encoding a 1×1 black image.
        # Keeps the rest of the system from hardcoding 512.
        with torch.no_grad():
            from PIL import Image as _Image
            probe = _Image.new("RGB", (8, 8), 0)
            x = self.preprocess(probe).unsqueeze(0).to(self.device)  # type: ignore[operator,union-attr]
            emb = self.model.encode_image(x)
            self.dim = int(emb.shape[-1])
        logger.info(
            f"[CLIPEncoder] {model_name}/{pretrained} loaded on "
            f"{self.device} (dim={self.dim})"
        )

    def encode_image(self, image_bgr: Optional[np.ndarray]) -> np.ndarray:
        """Encode a BGR numpy image (e.g. a YOLO crop) to an L2-
        normalized float32 vector. Empty / None input → zero vector."""
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        from PIL import Image
        rgb = image_bgr[..., ::-1]
        try:
            pil = Image.fromarray(rgb)
        except Exception as e:
            logger.debug(f"[CLIPEncoder] PIL conversion failed: {e}")
            return np.zeros(self.dim, dtype=np.float32)
        with self._torch.no_grad():
            x = self.preprocess(pil).unsqueeze(0).to(self.device)  # type: ignore[operator,union-attr]
            emb = self.model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # type: ignore[operator]
            return emb.cpu().numpy().squeeze(0).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query to an L2-normalized float32 vector."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        with self._torch.no_grad():
            toks = self.tokenizer([text]).to(self.device)  # type: ignore[operator]
            emb = self.model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # type: ignore[operator]
            return emb.cpu().numpy().squeeze(0).astype(np.float32)

    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._torch.no_grad():
            toks = self.tokenizer(texts).to(self.device)  # type: ignore[operator]
            emb = self.model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # type: ignore[operator]
            return emb.cpu().numpy().astype(np.float32)


class NullCLIPEncoder:
    """No-op encoder. encode_image / encode_text always return None,
    which the consumers (ObservationBuilder, _object_pair_cost,
    find_object) treat as 'no embedding available' and skip the
    visual signal cleanly. Stops the system from crashing on a
    missing open_clip while letting the rest of the pipeline run."""

    dim: int = 0

    def encode_image(self, image_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        return None

    def encode_text_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        return None


def try_load(
    model_name: str = CLIPEncoder.DEFAULT_MODEL,
    pretrained: str = CLIPEncoder.DEFAULT_PRETRAINED,
    device: str = "cuda",
) -> object:
    """Try to construct a real CLIPEncoder; on any failure (missing
    dep, weights download issue, CUDA unavailable, etc.) log a warning
    and return a NullCLIPEncoder so the rest of the system can
    construct without conditionals."""
    try:
        return CLIPEncoder(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
    except Exception as e:
        logger.warning(
            f"[CLIPEncoder] could not load — object dedup + find_object "
            f"will degrade to no-op. Install with `pip install "
            f"open_clip_torch` to enable. Error: {e}"
        )
        return NullCLIPEncoder()
