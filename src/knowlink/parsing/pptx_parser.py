from __future__ import annotations

import io
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .base import ImageRef, ParseResult, ParserCapabilityError
from .registry import OLE_MAGIC, register


def _parse_pptx_markitdown(data: bytes) -> tuple[str, list[str]]:
    from markitdown import MarkItDown

    md = MarkItDown()
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        f.write(data)
        f.flush()
        try:
            result = md.convert(f.name)
            return (result.text_content or "", [])
        except Exception:
            return ("", [])


def _extract_media_from_zip(data: bytes) -> list[ImageRef]:
    images: list[ImageRef] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        media_files = sorted(
            n for n in zf.namelist() if n.startswith("ppt/media/")
        )
        for name in media_files:
            raw = zf.read(name)
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")):
                images.append(ImageRef(data=raw, mime="image/png"))
            elif lower.endswith((".wmf", ".emf", ".svg")):
                png_data = _rasterize_vector(raw, lower)
                if png_data:
                    images.append(ImageRef(data=png_data, mime="image/png"))
    except zipfile.BadZipFile:
        pass
    return images


def _rasterize_vector(raw: bytes, ext: str) -> bytes | None:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        pass
    return _rasterize_with_imagemagick(raw, ext)


def _rasterize_with_imagemagick(raw: bytes, ext: str) -> bytes | None:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"input{ext}"
            src.write_bytes(raw)
            dst = Path(tmp) / "output.png"
            subprocess.run(
                ["magick", "convert", str(src), "-density", "150", str(dst)],
                capture_output=True,
                timeout=30,
            )
            if dst.exists() and dst.stat().st_size > 100:
                return dst.read_bytes()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _fix_unresolved_images(markdown: str, images: list[ImageRef]) -> tuple[str, list[ImageRef]]:
    unresolved_re = re.compile(r'!\[([^\]]*)\]\((?!data:|http|images/)([^)]+)\)')
    used: list[ImageRef] = []
    img_iter = iter(images)

    def replacer(m: re.Match) -> str:
        try:
            ref = next(img_iter)
            used.append(ref)
            return f"![{m.group(1)}](images/{ref.id}.png)"
        except StopIteration:
            return m.group(0)

    fixed = unresolved_re.sub(replacer, markdown)
    remaining = list(img_iter)
    used.extend(remaining)
    return fixed, used


@register("pptx")
def parse_pptx(data: bytes, filename: str, **kwargs) -> ParseResult:
    if data[:8] == OLE_MAGIC:
        return parse_ppt(data, filename)

    markdown, _ = _parse_pptx_markitdown(data)
    media = _extract_media_from_zip(data)

    if markdown:
        markdown, used_images = _fix_unresolved_images(markdown, media)
        return ParseResult(markdown=markdown, images=used_images)

    if media:
        parts = [f"![slide_{i}](images/{ref.id}.png)" for i, ref in enumerate(media)]
        return ParseResult(markdown="\n\n".join(parts), images=media)

    return ParseResult(markdown="")


@register("ppt")
def parse_ppt(data: bytes, filename: str, **kwargs) -> ParseResult:
    from .docx_parser import _convert_with_libreoffice, _has_libreoffice

    converted = _convert_with_libreoffice(data, ".ppt", ".pptx")
    if converted is not None:
        return parse_pptx(converted, filename)

    if not _has_libreoffice():
        raise ParserCapabilityError("ppt", "LibreOffice (soffice)")
    raise ParserCapabilityError("ppt", "LibreOffice (soffice)")
