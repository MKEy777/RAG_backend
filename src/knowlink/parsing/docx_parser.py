from __future__ import annotations

import io
import re
import subprocess
import tempfile
from pathlib import Path

from .base import ImageRef, ParseResult, ParserCapabilityError
from .registry import OLE_MAGIC, register

# 中文编号标题模式
_NUM_H1_RE = re.compile(r"^\d+\s+\S")
_NUM_H2_RE = re.compile(r"^\d+\.\d+\s+\S")
_NUM_H3_RE = re.compile(r"^\d+\.\d+\.\d+\s+\S")
_CN_H1_RE = re.compile(r"^[一二三四五六七八九十]+、")
_HEADING_MAX_LEN = 40


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
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        src.write_bytes(data)
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", target_suffix.lstrip("."), "--outdir", tmp, str(src)],
                capture_output=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        out = Path(tmp) / f"input{target_suffix}"
        if out.exists():
            return out.read_bytes()
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


def _parse_docx_python_docx(data: bytes) -> ParseResult:
    from docx import Document
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))

    # rId -> ImageRef 映射
    img_map: dict[str, ImageRef] = {}
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.reltype:
            try:
                img_map[rel_id] = ImageRef(data=rel.target_part.blob, mime="image/png")
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
            rows_text = []
            for row in block.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows_text.append("| " + " | ".join(cells) + " |")
            if rows_text:
                header_sep = "| " + " | ".join(["---"] * len(block.rows[0].cells)) + " |"
                rows_text.insert(1, header_sep)
                parts.append("\n".join(rows_text))

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
