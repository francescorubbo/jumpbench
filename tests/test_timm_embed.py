from __future__ import annotations

import numpy as np
import pytest

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.embed.generate import _embed_tiles, _prepare_image, crop_site, embed_site


def test_timm_prepare_image_skips_site_preprocess():
    card = resolve_model("timm")
    image = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    prepared, names = _prepare_image(image, card)
    assert names == ["AGP", "DNA", "ER", "Mito", "RNA"]
    np.testing.assert_array_equal(prepared, image)


def test_embed_tiles_applies_tile_preprocess_before_resize():
    card = resolve_model("timm")
    card["tile_size"] = 8
    card["crop_size"] = 8
    dim = np.linspace(0.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    bright = np.linspace(500.0, 600.0, 64, dtype=np.float32).reshape(8, 8)
    image = np.zeros((5, 8, 16), dtype=np.float32)
    image[:, :, :8] = dim
    image[:, :, 8:] = bright

    prepared, _ = _prepare_image(image, card)
    np.testing.assert_allclose(prepared, image)

    seen: list[np.ndarray] = []

    class _Rec:
        embedding_dim = 4
        resize_to = 4

        def embed_tiles(self, tiles: np.ndarray) -> np.ndarray:
            seen.append(tiles.copy())
            return np.zeros((tiles.shape[0], 4), dtype=np.float32)

    tiles, _, _ = crop_site(image, card)
    assert tiles.shape == (2, 5, 8, 8)
    feats = _embed_tiles(_Rec(), tiles, card)
    assert feats.shape == (2, 4)
    assert len(seen) == 1
    out = seen[0]
    assert out.shape == (2, 5, 4, 4)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert float(out[0].mean()) == pytest.approx(float(out[1].mean()), abs=0.15)


def test_timm_resnet_bag_of_channels_keeps_native_size():
    pytest.importorskip("timm")
    pytest.importorskip("torch")
    cfg = apply_overrides(
        load_models_config(),
        [
            "models.timm.architecture=resnet18",
            "models.timm.pretrained=false",
            "runtime.device=cpu",
            "runtime.batch_size=2",
        ],
    )
    card = resolve_model("timm", cfg)
    card["channels"] = ["AGP", "DNA"]
    from jumpbench.embed.backends import TimmBackend

    backend = TimmBackend(card)
    assert backend.resize_to is None
    tiles = np.random.default_rng(0).random((3, 2, 32, 40)).astype(np.float32)
    out = backend.embed_tiles(tiles)
    assert out.shape == (3, 2 * int(backend.model.num_features))
    assert np.isfinite(out).all()


def test_timm_vit_resizes_via_embed_tiles():
    pytest.importorskip("timm")
    pytest.importorskip("torch")
    cfg = apply_overrides(
        load_models_config(),
        [
            "models.timm.architecture=vit_tiny_patch16_224",
            "models.timm.pretrained=false",
            "models.timm.tile_size=32",
            "runtime.device=cpu",
            "runtime.batch_size=2",
        ],
    )
    card = resolve_model("timm", cfg)
    card["channels"] = ["AGP"]
    from jumpbench.embed.backends import TimmBackend

    backend = TimmBackend(card)
    assert backend.resize_to == 32
    rng = np.random.default_rng(1)
    image = rng.integers(0, 1024, size=(5, 48, 48)).astype(np.uint16)
    card["crop_size"] = 48
    feats, extra, _ = embed_site(image, card, backend)
    assert feats.shape == (1, int(backend.model.num_features))
    assert extra["tile_y"].shape[0] == 1
    assert np.isfinite(feats).all()


def test_torch_device_auto_prefers_cuda_then_mps():
    pytest.importorskip("torch")
    import torch

    from jumpbench.embed.backends import _torch_device

    assert _torch_device("cpu").type == "cpu"
    monkey_cuda = torch.cuda.is_available()
    if monkey_cuda:
        assert _torch_device("auto").type == "cuda"
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        assert _torch_device("auto").type == "mps"
    else:
        assert _torch_device("auto").type == "cpu"
