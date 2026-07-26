from __future__ import annotations

import base64
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

from .base import ImageRef, ParseResult
from .registry import register

# base64 内嵌图片: ![alt](data:image/png;base64,...)
_B64_IMG_RE = re.compile(
    r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"
)
# HTML <img> 标签
_HTML_IMG_RE = re.compile(
    r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*/?>',
    re.IGNORECASE,
)
# 表格对齐行
_TABLE_SEP_RE = re.compile(r"^:?-{3,}:?$")
# markdown 图片引用: ![alt](path)
_MD_IMG_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_MIME_MAP = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}


def _extract_base64_images(text: str) -> tuple[str, list[ImageRef]]:
    """Extract base64 embedded images, replace with placeholder refs."""
    images: list[ImageRef] = []

    def repl(m: re.Match) -> str:
        alt = m.group(1)
        fmt = m.group(2).lower()
        b64_data = m.group(3)
        try:
            raw = base64.b64decode(b64_data)
        except Exception:
            return alt or ""
        if len(raw) < 100:
            return alt or ""
        mime = f"image/{fmt}" if fmt in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
        ref = ImageRef(data=raw, mime=mime)
        images.append(ref)
        return f"![{alt}](images/{ref.id}.{fmt})"

    text = _B64_IMG_RE.sub(repl, text)
    return text, images


def _normalize_html_img(text: str) -> str:
    """Convert <img src="..."> tags to markdown image syntax."""

    def repl(m: re.Match) -> str:
        src = m.group(1)
        alt_m = re.search(r'alt=["\']([^"\']*)["\']', m.group(0), re.IGNORECASE)
        alt = alt_m.group(1) if alt_m else ""
        return f"![{alt}]({src})"

    return _HTML_IMG_RE.sub(repl, text)


def _resolve_local_images(text: str, images: list[ImageRef], base_dir: str) -> str:
    """Resolve relative image paths from local filesystem (base_dir mode)."""
    base = Path(base_dir)
    if not base.is_dir():
        return text

    def repl(m: re.Match) -> str:
        alt = m.group(1)
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:", "images/")):
            return m.group(0)
        ext = PurePosixPath(src).suffix.lower()
        if ext not in _IMG_EXTENSIONS:
            return m.group(0)
        # 尝试相对路径拼接
        img_path = base / src.replace("/", "\\") if "\\" in src else base / src
        if not img_path.is_file():
            # 回退：只取文件名在 base_dir 下查找
            img_path = base / Path(src).name
        if not img_path.is_file():
            return m.group(0)
        try:
            img_data = img_path.read_bytes()
        except OSError:
            return m.group(0)
        if len(img_data) < 100:
            return m.group(0)
        mime = _MIME_MAP.get(ext, "image/png")
        ref = ImageRef(data=img_data, mime=mime)
        images.append(ref)
        return f"![{alt}](images/{ref.id}{ext})"

    return _MD_IMG_REF_RE.sub(repl, text)


def _format_tables(text: str) -> str:
    """Normalize markdown table spacing for consistent chunking."""
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(_TABLE_SEP_RE.match(c) for c in cells if c):
                normalized = []
                for c in cells:
                    left = ":" if c.startswith(":") else ""
                    right = ":" if c.endswith(":") else ""
                    normalized.append(f"{left}---{right}")
                result.append("| " + " | ".join(normalized) + " |")
            else:
                result.append("| " + " | ".join(cells) + " |")
        else:
            result.append(line)
    return "\n".join(result)


@register("md")
def parse_markdown(data: bytes, filename: str, **kwargs) -> ParseResult:
    base_dir = kwargs.get("base_dir")
    text = data.decode("utf-8", errors="replace")
    text, images = _extract_base64_images(text)
    text = _normalize_html_img(text)
    if base_dir:
        text = _resolve_local_images(text, images, base_dir)
    text = _format_tables(text)
    return ParseResult(markdown=text, images=images)


@register("txt")
def parse_text(data: bytes, filename: str, **kwargs) -> ParseResult:
    return ParseResult(markdown=data.decode("utf-8", errors="replace"))


@register("markdown")
def parse_markdown_alt(data: bytes, filename: str, **kwargs) -> ParseResult:
    return parse_markdown(data, filename, **kwargs)


@register("zip")
def parse_zip(data: bytes, filename: str, **kwargs) -> ParseResult:
    """Parse a zip containing a markdown file + referenced image assets."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return ParseResult(markdown="")

    names = zf.namelist()

    md_name = None
    for n in names:
        if n.lower().endswith((".md", ".markdown")):
            md_name = n
            break
    if md_name is None:
        texts = []
        for n in names:
            if n.lower().endswith((".txt", ".md", ".csv")):
                texts.append(zf.read(n).decode("utf-8", errors="replace"))
        return ParseResult(markdown="\n\n".join(texts))

    text = zf.read(md_name).decode("utf-8", errors="replace")
    md_dir = str(PurePosixPath(md_name).parent)

    text, images = _extract_base64_images(text)
    text = _normalize_html_img(text)

    def resolve_img(m: re.Match) -> str:
        alt = m.group(1)
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:", "images/")):
            return m.group(0)
        if md_dir and md_dir != ".":
            candidate = f"{md_dir}/{src}"
        else:
            candidate = src
        candidate = str(PurePosixPath(candidate))
        img_data = None
        for n in names:
            if n.replace("\\", "/") == candidate or n == src:
                img_data = zf.read(n)
                break
        if img_data is None:
            return m.group(0)
        ext = PurePosixPath(src).suffix.lower()
        if ext not in _IMG_EXTENSIONS:
            return m.group(0)
        if len(img_data) < 100:
            return m.group(0)
        mime = _MIME_MAP.get(ext, "image/png")
        ref = ImageRef(data=img_data, mime=mime)
        images.append(ref)
        return f"![{alt}](images/{ref.id}{ext})"

    text = _MD_IMG_REF_RE.sub(resolve_img, text)
    text = _format_tables(text)
    return ParseResult(markdown=text, images=images)
