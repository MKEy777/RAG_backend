from __future__ import annotations

from knowlink.parsing.registry import detect_format, parse_file
from knowlink.parsing.base import ParseResult


def test_detect_pdf():
    data = b"%PDF-1.4 fake content"
    assert detect_format(data, "test.pdf") == "pdf"


def test_detect_ole_docx():
    data = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1" + b"\x00" * 100
    assert detect_format(data, "test.docx") == "doc"


def test_detect_zip_docx():
    data = b"PK\x03\x04" + b"\x00" * 100
    assert detect_format(data, "test.docx") == "docx"


def test_parse_markdown():
    data = b"# Hello\n\nWorld"
    result = parse_file(data, "test.md")
    assert isinstance(result, ParseResult)
    assert "# Hello" in result.markdown
    assert result.images == []


def test_parse_txt():
    data = "Plain text content".encode()
    result = parse_file(data, "test.txt")
    assert result.markdown == "Plain text content"
