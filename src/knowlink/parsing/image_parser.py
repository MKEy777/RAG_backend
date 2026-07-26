from __future__ import annotations

from .base import ImageRef, ParseResult
from .registry import register

_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "webp": "image/webp", "bmp": "image/bmp", "gif": "image/gif",
}


def _parse_image(data: bytes, filename: str, **kwargs) -> ParseResult:
    ext = filename.rsplit(".", 1)[-1].lower()
    ref = ImageRef(data=data, mime=_MIME.get(ext, "image/png"))
    return ParseResult(markdown=f"![{filename}](images/{ref.id}.{ext})", images=[ref])


for _ext in _MIME:
    register(_ext)(_parse_image)
