from __future__ import annotations

import io
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .base import ImageRef, ParseResult, ParserCapabilityError
from .registry import OLE_MAGIC, register

_NUM_H1_RE = re.compile(r"^\d+\s+\S")
_NUM_H2_RE = re.compile(r"^\d+\.\d+\s+\S")
_NUM_H3_RE = re.compile(r"^\d+\.\d+\.\d+\s+\S")
_CN_H1_RE = re.compile(r"^[一二三四五六七八九十]+、")
_HEADING_MAX_LEN = 40
_LO_RETRIES = 3
_LO_BACKOFF_BASE = 0.5
_IMG_MIN_PX = 50
_IMG_MAX_EDGE = 1920


def _has_libreoffice() -> bool:
    try:
        subprocess.run(
            ["soffice", "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _convert_with_libreoffice(data: bytes, suffix: str, target_suffix: str) -> bytes | None:
    if not _has_libreoffice():
        return None
    for attempt in range(_LO_RETRIES):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"input{suffix}"
            src.write_bytes(data)
            profile_dir = Path(tmp) / "lo_profile"
            profile_dir.mkdir()
            profile_url = profile_dir.as_uri()
            try:
                subprocess.run(
                    [
                        "soffice", "--headless",
                        f"-env:UserInstallation={profile_url}",
                        "--convert-to", target_suffix.lstrip("."),
                        "--outdir", tmp, str(src),
                    ],
                    capture_output=True,
                    timeout=120,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                if attempt < _LO_RETRIES - 1:
                    time.sleep(_LO_BACKOFF_BASE * (attempt + 1))
                    continue
                return None
            out = Path(tmp) / f"input{target_suffix}"
            if out.exists():
                return out.read_bytes()
        if attempt < _LO_RETRIES - 1:
            time.sleep(_LO_BACKOFF_BASE * (attempt + 1))
    return None


def _parse_docx_markitdown(data: bytes) -> ParseResult | None:
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(data)
            f.flush()
            result = md.convert(f.name)
        text = result.text_content if result else ""
        if not text or not text.strip():
            return None
        return ParseResult(markdown=text)
    except Exception:
        return None



# patch: python-docx issue #1105 — 某些 DOCX 含 target_ref='../NULL' 导致崩溃
def _patch_python_docx():
    try:
        from docx.opc.pkgreader import _SerializedRelationships, _SerializedRelationship
        from docx.opc.oxml import parse_xml

        def _load_from_xml_v2(baseURI, rels_item_xml):
            srels = _SerializedRelationships()
            if rels_item_xml is not None:
                rels_elm = parse_xml(rels_item_xml)
                for rel_elm in rels_elm.Relationship_lst:
                    if rel_elm.target_ref in ('../NULL', 'NULL'):
                        continue
                    srels._srels.append(_SerializedRelationship(baseURI, rel_elm))
            return srels

        _SerializedRelationships.load_from_xml = _load_from_xml_v2
    except Exception:
        pass


_patch_python_docx()


def _detect_heading_level(text: str, style_name: str) -> int:
    """Return heading level (1-3) or 0 if not a heading."""
    if style_name.startswith("heading"):
        try:
            return int(style_name.replace("heading", "").strip())
        except ValueError:
            return 1
    if len(text) > _HEADING_MAX_LEN:
        return 0
    if _NUM_H3_RE.match(text):
        return 3
    if _NUM_H2_RE.match(text):
        return 2
    if _CN_H1_RE.match(text):
        return 1
    if _NUM_H1_RE.match(text):
        return 1
    return 0


def _extract_omml(paragraph) -> str:
    """Extract OMML formulas from a paragraph and return LaTeX approximation."""
    from docx.oxml.ns import qn

    formulas = []
    for omath in paragraph._element.iter(qn('m:oMath')):
        texts = []
        for r in omath.iter(qn('m:r')):
            t = r.find(qn('m:t'))
            if t is not None and t.text:
                texts.append(t.text)
        if texts:
            formulas.append("".join(texts))
    if not formulas:
        return ""
    return " ".join(f"${f}$" for f in formulas)


def _filter_and_resize_image(blob: bytes) -> ImageRef | None:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(blob))
        w, h = img.size
        if w < _IMG_MIN_PX or h < _IMG_MIN_PX:
            return None
        max_edge = max(w, h)
        if max_edge > _IMG_MAX_EDGE:
            ratio = _IMG_MAX_EDGE / max_edge
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return ImageRef(data=buf.getvalue(), mime="image/png")
    except Exception:
        if len(blob) >= 100:
            return ImageRef(data=blob, mime="image/png")
        return None


def _convert_table_to_markdown(table) -> str:
    from docx.oxml.ns import qn

    rows_data: list[list[str]] = []
    for row in table.rows:
        cells_text: list[str] = []
        prev_text = None
        for cell in row.cells:
            text = cell.text.strip().replace("\n", " ")
            tc = cell._tc
            grid_span = tc.find(qn('w:tcPr'))
            span = 1
            if grid_span is not None:
                gs = grid_span.find(qn('w:gridSpan'))
                if gs is not None:
                    try:
                        span = int(gs.get(qn('w:val'), '1'))
                    except (ValueError, TypeError):
                        span = 1
            if text == prev_text and span > 1:
                continue
            cells_text.append(text)
            prev_text = text
        rows_data.append(cells_text)

    if not rows_data:
        return ""
    max_cols = max(len(r) for r in rows_data)
    for r in rows_data:
        while len(r) < max_cols:
            r.append("")
    lines = []
    for i, row in enumerate(rows_data):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    return "\n".join(lines)


def _parse_docx_python_docx(data: bytes) -> ParseResult:
    from docx import Document
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))

    img_map: dict[str, ImageRef] = {}
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.reltype:
            try:
                blob = rel.target_part.blob
                ref = _filter_and_resize_image(blob)
                if ref:
                    img_map[rel_id] = ref
            except Exception:
                pass

    parts: list[str] = []
    images: list[ImageRef] = []

    def _iter_block_items(parent):
        from docx.oxml.ns import qn
        body = parent.element.body
        for child in body:
            if child.tag == qn('w:p'):
                yield Paragraph(child, parent)
            elif child.tag == qn('w:tbl'):
                yield DocxTable(child, parent)

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            style_name = (block.style.name or "").lower()

            # 跳过目录段落
            if "toc" in style_name or "目录" in style_name:
                continue

            # 检查段落内图片
            blips = block._element.xpath('.//a:blip/@r:embed')
            for rid in blips:
                ref = img_map.get(rid)
                if ref:
                    images.append(ref)
                    parts.append(f"![img_{len(images)}](images/{ref.id}.png)")

            text = block.text.strip()
            if not text:
                continue

            # 提取 OMML 公式
            formula = _extract_omml(block)

            # 标题检测：Word 内置样式 + 中文编号模式
            level = _detect_heading_level(text, style_name)
            if level > 0:
                parts.append(f"{'#' * level} {text}")
            elif formula:
                parts.append(f"{text}\n\n{formula}")
            else:
                parts.append(text)
        elif isinstance(block, DocxTable):
            rows_text = _convert_table_to_markdown(block)
            if rows_text:
                parts.append(rows_text)

    return ParseResult(markdown="\n\n".join(parts), images=images)


def _extract_docx_ordered(data: bytes) -> ParseResult:
    """python-docx 顺序提取（文字+图片交错），作为 markitdown 的补充图片源。"""
    try:
        return _parse_docx_python_docx(data)
    except Exception:
        return ParseResult(markdown="")


@register("docx")
def parse_docx(data: bytes, filename: str, **kwargs) -> ParseResult:
    if data[:8] == OLE_MAGIC:
        return parse_doc(data, filename)

    # 始终用 python-docx 顺序提取（保留图片位置）
    return _parse_docx_python_docx(data)


@register("doc")
def parse_doc(data: bytes, filename: str, **kwargs) -> ParseResult:
    converted = _convert_with_libreoffice(data, ".doc", ".docx")
    if converted is not None:
        result = _parse_docx_markitdown(converted)
        if result is not None:
            return result
        return _parse_docx_python_docx(converted)

    if not _has_libreoffice():
        raise ParserCapabilityError("doc", "LibreOffice (soffice)")

    raise ParserCapabilityError("doc", "LibreOffice (soffice)")
