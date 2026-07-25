from __future__ import annotations

import io

from PIL import Image

from .base import ImageRef


def to_jpeg(data: bytes, quality: int = 85) -> bytes:
    img = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_png(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_dimensions(data: bytes) -> tuple[int, int]:
    img = Image.open(io.BytesIO(data))
    return img.size


def make_image_ref(data: bytes, mime: str = "image/jpeg") -> ImageRef:
    return ImageRef(data=data, mime=mime)
