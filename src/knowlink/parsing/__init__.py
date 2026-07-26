from __future__ import annotations

from .base import ImageRef, ParseResult, ParserCapabilityError
from .registry import detect_format, parse_file

from . import pdf_parser  # noqa: F401
from . import docx_parser  # noqa: F401
from . import excel_parser  # noqa: F401
from . import pptx_parser  # noqa: F401
from . import text_parser  # noqa: F401

__all__ = [
    "ImageRef",
    "ParseResult",
    "ParserCapabilityError",
    "detect_format",
    "parse_file",
]
