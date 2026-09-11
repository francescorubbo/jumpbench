import numpy as np

from jumpbench.config import apply_overrides, load_models_config, resolve_model
from jumpbench.embed.backends import DummyBackend
from jumpbench.embed.generate import embed_site


def test_dummy_embed_is_deterministic():
    card = resolve_model(
        "dummy", apply_overrides(load_models_config(), ["models.dummy.tile_size=32"])
    )
    backend = DummyBackend(card)
    image = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
    a, extra, _stats = embed_site(image, card, backend)
    b, extra_b, _ = embed_site(image, card, backend)
    assert a.shape[0] == 4  # 64/32 * 64/32
    assert extra["tile_y"].shape[0] == 4
    assert np.allclose(a, b)
    assert np.array_equal(extra["tile_x"], extra_b["tile_x"])
