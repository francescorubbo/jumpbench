"""On-disk image codecs. Default is paper JPEG XL MQ so we never persist TIFFs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Paper JUMP-lite MQ is ~116 GB vs ~10 TB raw (≈ 1.2%). HQ ≈ 2.4%.
PAPER_MQ_RATIO = 0.012
PAPER_HQ_RATIO = 0.024


@dataclass(frozen=True)
class ImageCodec:
    name: str
    suffix: str
    kind: str  # raw | jpegxl | jpeg
    distance: float | None = None
    jpeg_quality: int | None = None
    min_complete_bytes: int = 128
    paper_ratio: float = 1.0
    description: str = ""


CODECS: dict[str, ImageCodec] = {
    "jpegxl_mq": ImageCodec(
        name="jpegxl_mq",
        suffix=".jxl",
        kind="jpegxl",
        distance=3.0,
        min_complete_bytes=24,
        paper_ratio=PAPER_MQ_RATIO,
        description="JPEG XL Butteraugli distance 3.0 (paper MQ, ~100× vs TIFF)",
    ),
    "jpegxl_hq": ImageCodec(
        name="jpegxl_hq",
        suffix=".jxl",
        kind="jpegxl",
        distance=1.0,
        min_complete_bytes=24,
        paper_ratio=PAPER_HQ_RATIO,
        description="JPEG XL distance 1.0 (paper HQ)",
    ),
    "jpeg": ImageCodec(
        name="jpeg",
        suffix=".jpg",
        kind="jpeg",
        jpeg_quality=90,
        min_complete_bytes=24,
        paper_ratio=0.02,
        description="8-bit JPEG quality 90 (lossy bit depth; not the paper codec)",
    ),
    "raw": ImageCodec(
        name="raw",
        suffix=".tif",
        kind="raw",
        min_complete_bytes=64 * 1024,
        paper_ratio=1.0,
        description="Uncompressed Orig TIFF (multiple TB; needs a large disk)",
    ),
}

DEFAULT_CODEC = "jpegxl_mq"


def get_codec(name: str) -> ImageCodec:
    if name not in CODECS:
        known = ", ".join(CODECS)
        raise ValueError(f"Unknown codec {name!r}. Known: {known}")
    return CODECS[name]


def _require_imagecodecs():
    try:
        import imagecodecs
    except ImportError as exc:
        raise ImportError(
            "Streaming JPEG XL compression needs imagecodecs. "
            "Install with: pip install imagecodecs"
        ) from exc
    return imagecodecs


def encode_image(array: np.ndarray, codec: ImageCodec | str) -> bytes:
    spec = get_codec(codec) if isinstance(codec, str) else codec
    if spec.kind == "raw":
        raise ValueError("raw codec is stored as TIFF, not encoded here")
    if spec.kind == "jpegxl":
        imagecodecs = _require_imagecodecs()
        return imagecodecs.jpegxl_encode(
            np.ascontiguousarray(array),
            photometric="gray",
            lossless=False,
            distance=float(spec.distance or 3.0),
        )
    if spec.kind == "jpeg":
        imagecodecs = _require_imagecodecs()
        if array.dtype != np.uint8:
            peak = max(float(array.max()), 1.0)
            array = np.clip(array.astype(np.float32) * (255.0 / peak), 0, 255).astype(np.uint8)
        return imagecodecs.jpeg_encode(
            np.ascontiguousarray(array),
            level=int(spec.jpeg_quality or 90),
        )
    raise ValueError(f"Cannot encode kind {spec.kind}")


def decode_image(payload: bytes, codec: ImageCodec | str | None = None) -> np.ndarray:
    imagecodecs = _require_imagecodecs()
    spec = get_codec(codec) if isinstance(codec, str) else codec
    if spec is None or spec.kind == "jpegxl":
        try:
            array = np.squeeze(np.asarray(imagecodecs.jpegxl_decode(payload)))
            return array
        except Exception:
            if spec is not None:
                raise
    return np.squeeze(np.asarray(imagecodecs.jpeg_decode(payload)))
