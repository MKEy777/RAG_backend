from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from .base import ParseResult, ParserCapabilityError
from .registry import register

DISPIMG_RE = re.compile(r'=_?xlfn\.(DISPIMG|IMAGE)\s*\(', re.IGNORECASE)


def _col_letter(idx: int) -> str:
    result = ""
    while idx >= 0:
        result = chr(65 + idx % 26) + result
        idx = idx // 26 - 1
    return result


def _repair_shared_strings(data: bytes) -> bytes:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = zf.namelist()
        if "xl/sharedStrings.xml" not in names:
            return data
        return data
    except zipfile.BadZipFile:
        return data


def _fill_merged_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill(axis=0).ffill(axis=1)


def _filter_dispimg(text: str) -> str:
    return DISPIMG_RE.sub("", text).strip()


def _parse_xlsx_data(data: bytes) -> ParseResult:
    data = _repair_shared_strings(data)
    try:
        xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception:
        from .docx_parser import _convert_with_libreoffice

        converted = _convert_with_libreoffice(data, ".xlsx", ".xlsx")
        if converted is None:
            raise ParserCapabilityError("xlsx", "LibreOffice (soffice)")
        xls = pd.ExcelFile(io.BytesIO(converted), engine="openpyxl")

    parts: list[str] = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, header=None)
        if df.empty:
            continue
        df = _fill_merged_cells(df)
        parts.append(f"## Sheet: {sheet_name}\n")
        for row_idx, row in df.iterrows():
            cells = []
            all_empty = True
            for col_idx, val in enumerate(row):
                if pd.isna(val):
                    continue
                text = _filter_dispimg(str(val))
                if not text:
                    continue
                all_empty = False
                cells.append(f"{_col_letter(col_idx)}: {text}")
            if all_empty:
                continue
            parts.append(", ".join(cells))

    return ParseResult(markdown="\n".join(parts))


@register("xlsx")
def parse_xlsx(data: bytes, filename: str) -> ParseResult:
    return _parse_xlsx_data(data)


@register("xls")
def parse_xls(data: bytes, filename: str) -> ParseResult:
    from .docx_parser import _convert_with_libreoffice

    converted = _convert_with_libreoffice(data, ".xls", ".xlsx")
    if converted is not None:
        return _parse_xlsx_data(converted)
    try:
        xls = pd.ExcelFile(io.BytesIO(data), engine="xlrd")
        parts: list[str] = []
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name, header=None)
            if df.empty:
                continue
            parts.append(f"## Sheet: {sheet_name}\n")
            for _, row in df.iterrows():
                cells = []
                all_empty = True
                for col_idx, val in enumerate(row):
                    if pd.isna(val):
                        continue
                    text = str(val).strip()
                    if not text:
                        continue
                    all_empty = False
                    cells.append(f"{_col_letter(col_idx)}: {text}")
                if all_empty:
                    continue
                parts.append(", ".join(cells))
        return ParseResult(markdown="\n".join(parts))
    except Exception:
        raise ParserCapabilityError("xls", "LibreOffice (soffice)")
