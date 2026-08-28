"""Model backends. Torch is imported lazily so unit tests can run without GPU stacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EmbeddingBackend(ABC):
    name: str
    embedding_dim: int | None = None

    def __init__(self, card: dict[str, Any]):
        self.card = card
        self.name = card["name"]

    @abstractmethod
    def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
        """tiles: (N, C, H, W) float32 -> (N, D) float32."""


class DummyBackend(EmbeddingBackend):
    """Deterministic features from tile statistics. No weights, tests only."""

    def __init__(self, card: dict[str, Any]):
        super().__init__(card)
        self.embedding_dim = int(card.get("embedding_dim", 32))
        self.seed = int(card.get("runtime", {}).get("seed", 0))

    def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
        n, c, h, w = tiles.shape
        rng = np.random.default_rng(self.seed)
        proj = rng.standard_normal((c * 4, self.embedding_dim)).astype(np.float32)
        stats = np.stack(
            [
                tiles.mean(axis=(2, 3)),
                tiles.std(axis=(2, 3)),
                tiles.reshape(n, c, -1).min(axis=2),
                tiles.reshape(n, c, -1).max(axis=2),
            ],
            axis=2,
        ).reshape(n, c * 4)
        return stats @ proj


def _torch_device(spec: str):
    import torch

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class DinoV2Backend(EmbeddingBackend):
    def __init__(self, card: dict[str, Any]):
        super().__init__(card)
        import torch

        device = _torch_device(card.get("runtime", {}).get("device", "auto"))
        self.device = device
        repo = card.get("checkpoint", "facebookresearch/dinov2")
        arch = card.get("architecture", "dinov2_vits14")
        pretrained = bool(card.get("pretrained", True))
        self.model = torch.hub.load(repo, arch, pretrained=pretrained)
        self.model.eval().to(device)
        self.batch_size = int(card.get("runtime", {}).get("batch_size", 16))

    def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
        import torch

        # DINOv2 is RGB. Microscopy 3-channel stacks are fed as-is (paper).
        x = torch.from_numpy(tiles.astype(np.float32, copy=False))
        outs = []
        with torch.inference_mode():
            for start in range(0, len(x), self.batch_size):
                batch = x[start : start + self.batch_size].to(self.device)
                feats = self.model(batch)
                if isinstance(feats, dict):
                    feats = feats.get("x_norm_clstoken", next(iter(feats.values())))
                outs.append(feats.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)


class MorphEmBackend(EmbeddingBackend):
    """Official MorphEm is single-channel bag-of-channels, then concatenate."""

    def __init__(self, card: dict[str, Any]):
        super().__init__(card)
        import torch
        from transformers import AutoModel

        device = _torch_device(card.get("runtime", {}).get("device", "auto"))
        self.device = device
        self.model = AutoModel.from_pretrained(card.get("checkpoint", "CaicedoLab/MorphEm"), trust_remote_code=True)
        self.model.eval().to(device)
        self.batch_size = int(card.get("runtime", {}).get("batch_size", 8))
        self.bag_of_channels = bool(card.get("bag_of_channels", True))

    def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(tiles.astype(np.float32, copy=False))
        channel_feats = []
        n_channels = x.shape[1]
        with torch.inference_mode():
            for c in range(n_channels):
                outs = []
                for start in range(0, len(x), self.batch_size):
                    ch = x[start : start + self.batch_size, c : c + 1].to(self.device)
                    output = self.model.forward_features(ch)
                    token = output["x_norm_clstoken"] if isinstance(output, dict) else output
                    outs.append(token.detach().cpu().numpy())
                channel_feats.append(np.concatenate(outs, axis=0))
        return np.concatenate(channel_feats, axis=1)


class OpenPhenomBackend(EmbeddingBackend):
    def __init__(self, card: dict[str, Any]):
        super().__init__(card)
        import torch
        from transformers import AutoModel

        device = _torch_device(card.get("runtime", {}).get("device", "auto"))
        self.device = device
        self.model = AutoModel.from_pretrained(
            card.get("checkpoint", "recursionpharma/OpenPhenom"),
            trust_remote_code=True,
        )
        self.model.eval().to(device)
        self.batch_size = int(card.get("runtime", {}).get("batch_size", 8))

    def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(tiles.astype(np.float32, copy=False))
        outs = []
        with torch.inference_mode():
            for start in range(0, len(x), self.batch_size):
                batch = x[start : start + self.batch_size].to(self.device)
                if hasattr(self.model, "forward_features"):
                    output = self.model.forward_features(batch)
                else:
                    output = self.model(batch)
                if isinstance(output, dict):
                    output = output.get("x_norm_clstoken", next(iter(output.values())))
                if hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state[:, 0]
                outs.append(output.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)


class SubCellBackend(EmbeddingBackend):
    def __init__(self, card: dict[str, Any]):
        super().__init__(card)
        try:
            from subcell_analysis.embedder import SubCellEmbedder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SubCell backend needs the SubCellPortable package. "
                "Install it separately, or point --set checkpoint=... at a local wrapper."
            ) from exc
        self.inner = SubCellEmbedder(
            model_type=card.get("checkpoint", "mae_contrast_supcon_model"),
            channels=card.get("model_channels", "rybg"),
        )
        self.batch_size = int(card.get("runtime", {}).get("batch_size", 4))

    def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.embed(tiles), dtype=np.float32)


BACKENDS = {
    "dummy": DummyBackend,
    "dinov2": DinoV2Backend,
    "morphem": MorphEmBackend,
    "openphenom": OpenPhenomBackend,
    "subcell": SubCellBackend,
}


def build_backend(card: dict[str, Any]) -> EmbeddingBackend:
    family = card.get("family", card["name"])
    if family == "dinov2_random":
        family = "dinov2"
    if family == "subcell_clip01":
        family = "subcell"
    if family not in BACKENDS:
        raise KeyError(f"No embedding backend for family {family!r}")
    return BACKENDS[family](card)
