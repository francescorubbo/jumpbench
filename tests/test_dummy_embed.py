from jumpbench.embed.backends import DummyBackend
from jumpbench.embed.generate import embed_site
from jumpbench.config import resolve_model, apply_overrides, load_models_config
import numpy as np


def test_dummy_embed_is_deterministic():
    card = resolve_model("dummy", apply_overrides(load_models_config(), ["models.dummy.tile_size=32"]))
    backend = DummyBackend(card)
    image = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
    a, _ = embed_site(image, card, backend)
    b, _ = embed_site(image, card, backend)
    assert a.shape[0] == 4  # 64/32 * 64/32
    assert np.allclose(a, b)
