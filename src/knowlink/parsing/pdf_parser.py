from __future__ import annotations

import io
import os
import re
import threading
from statistics import median

import pypdfium2 as pdfium

from .base import ImageRef, ParseResult
from .registry import register

_PDFIUM_LOCK = threading.Lock()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# detail.md §pdf
SCAN_IMAGE_AREA_RATIO = _env_float("KNOWLINK_PDF_SCAN_IMAGE_RATIO", 0.5)
SCAN_MIN_CHARS = _env_int("KNOWLINK_PDF_SCAN_MIN_CHARS", 10)
SCAN_IMAGE_AREA_RATIO_LOW = 0.1
TITLE_H1_RATIO = 2.0
TITLE_H2_RATIO = 1.6
TITLE_H3_RATIO = 1.35
TITLE_NOISE_THRESHOLD = 0.4
TITLE_MIN_LEN = 2
TITLE_MAX_LEN = 80
HEADER_FOOTER_PAGE_RATIO = 0.6
HEADER_FOOTER_MIN_PAGES = 4
EMBED_IMG_MIN_PX = _env_int("KNOWLINK_PDF_EMBED_MIN_PIXELS", 80)
EMBED_IMG_AREA_RATIO = _env_float("KNOWLINK_PDF_EMBED_MIN_AREA_RATIO", 0.01)
EMBED_IMG_MAX_COUNT = _env_int("KNOWLINK_PDF_EMBED_MAX_IMAGES", 50)
EMBED_IMG_MAX_EDGE = 1600
EMBED_REPEAT_PAGE_FRAC = _env_float("KNOWLINK_PDF_EMBED_REPEAT_PAGE_FRAC", 0.5)
VECTOR_FIG_MIN_PATHS = 10
VECTOR_FIG_MIN_AREA = _env_float("KNOWLINK_PDF_MIN_CHART_REGION_AREA", 0.015)
VECTOR_FIG_MAX_AREA = _env_float("KNOWLINK_PDF_MAX_CHART_REGION_AREA", 0.42)
QUALITY_SINGLE_CHAR_RATIO = 0.18
QUALITY_PUNCT_RATIO = 0.12
QUALITY_DISORDER_RATIO = 0.20
QUALITY_ORIGINAL_DISORDER = 0.08
NOISE_COL_WIDTH_RATIO = _env_float("KNOWLINK_PDF_MARGIN_COL_WIDTH_RATIO", 0.12)
SPACE_GAP_RATIO = _env_float("KNOWLINK_PDF_WORD_GAP_WIDTH_RATIO", 0.4)
TITLE_END_PUNCT = set("。？！，；：.?!;,")
LINENUM_RUN_MIN = 3
FORCE_SCANNED_PDF = _env_bool("KNOWLINK_PDF_FORCE_SCANNED", False)

_CN_H1_RE = re.compile(r"^[一二三四五六七八九十]+、")
_CN_H2_RE = re.compile(r"^\d+\.\d+\s")
_CN_H3_RE = re.compile(r"^(?:步骤\d+|Q\d+|第\d+页摘要)")
_CODE_LINE_RE = re.compile(
    r"^\s*(?:class |def |import |from |return |self\.|if |elif |else:|for |while |"
    r"assert |raise |with |try:|except|super\(|nn\.|torch\.|np\.|print\(|#)"
)
_CODE_CONT_RE = re.compile(
    r"^\s*(?:[\w\.\[\]\,\s]+[=\(\[]|"
    r"[\w\.]+\.[\w]+\(|"
    r"[\)\]\},;]|"
    r"[A-Z]\w*\(|"
    r"\"\"\"|'''|"
    r"[\w\.]+\s*\,\s*[\w\.])"
)
_CN_SENTENCE_END_RE = re.compile(r"[。？！；：,.!?;:]$")
_CN_PARA_BREAK_RE = re.compile(r"[。？！；：]$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*•]\s|\d+[.)、]\s)")
_HAS_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_HAS_CODE_FEATURE_RE = re.compile(r"[=\(\)\[\]{}]|\w+\.\w+|self\.|:\s*$|,\s*$")

# detail.md §pdf: PDF文本层伪影字符（U+FFFE/软连字符/零宽字符）
_PDF_ARTIFACT_RE = re.compile(r"[\u00ad\u200b-\u200f\ufeff\ufffe\uffff]")
_PDF_ARTIFACT_JOIN_RE = re.compile(r"(\w)[\u00ad\ufffe](\w)")
_CHART_DEBRIS_LINE_RE = re.compile(
    r"^(?:"
    r"[\d\s.]+|"
    r"\d{1,2}|"
    r"\d+-layer|"
    r"iter\.\s*\(1e4\)|"
    r"(?:training|test)\s+error\s*\(%\)"
    r")$",
    re.IGNORECASE,
)
_CHART_LAYER_RE = re.compile(r"^\d+-layer$", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+\b", re.IGNORECASE)


def _is_scan_page(page: pdfium.PdfPage) -> bool:
    width, height = page.get_size()
    page_area = width * height
    if page_area == 0:
        return False
    textpage = page.get_textpage()
    char_count = textpage.count_chars()
    img_area = 0.0
    for obj in page.get_objects(filter=(3,)):  # 3 = FPDF_PAGEOBJ_IMAGE in pypdfium2 5.x
        m = obj.get_matrix()
        img_area += abs(m.a * m.d)
    ratio = img_area / page_area
    if ratio > SCAN_IMAGE_AREA_RATIO:
        return True
    if char_count < SCAN_MIN_CHARS and ratio > SCAN_IMAGE_AREA_RATIO_LOW:
        return True
    return False


# PPT 转 PDF 的公式页面检测：横版 + 数学符号密集
_MATH_SYMBOL_RE = re.compile(r"[\(\)\[\]{}\u2211\u220F\u222B\u2202\u2207\u221A\u221E\u00B1\u00D7\u00B7/^_=]")


def _is_formula_page(page: pdfium.PdfPage, extracted_text: str) -> bool:
    """Detect landscape pages with dense math symbols (PPT-converted PDFs)."""
    width, height = page.get_size()
    if width <= height:
        return False
    if not extracted_text or len(extracted_text) < 50:
        return False
    math_chars = len(_MATH_SYMBOL_RE.findall(extracted_text))
    total = len(extracted_text)
    return math_chars / total > 0.15


RENDER_MAX_EDGE = 3000  # detail.md §pdf: 渲染图长边上限


def _effective_scale(page: pdfium.PdfPage, base_scale: float = 2.0) -> float:
    width, height = page.get_size()
    long_edge = max(width, height) * base_scale
    if long_edge > RENDER_MAX_EDGE:
        return RENDER_MAX_EDGE / max(width, height)
    return base_scale


def _render_page_image(page: pdfium.PdfPage, scale: float = 2.0) -> bytes:
    scale = _effective_scale(page, scale)
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _collect_invisible_boxes(page: pdfium.PdfPage) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    try:
        raw = pdfium.raw
        for obj in page.get_objects():
            if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                continue
            try:
                mode = raw.FPDFTextObj_GetTextRenderMode(obj.raw)
            except Exception:
                continue
            if mode != raw.FPDF_TEXTRENDERMODE_INVISIBLE:
                continue
            try:
                left, bottom, right, top = obj.get_bounds()
            except Exception:
                continue
            boxes.append(
                (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
            )
    except Exception:
        return []
    return boxes


def _point_in_boxes(x: float, y: float, boxes: list[tuple[float, float, float, float]]) -> bool:
    for x0, y0, x1, y1 in boxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _extract_text_page(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    char_count = textpage.count_chars()
    if char_count == 0:
        return ""

    original_text = textpage.get_text_range(0, char_count)

    width, height = page.get_size()
    invisible_boxes = _collect_invisible_boxes(page)

    chars: list[dict] = []
    for i in range(char_count):
        c = textpage.get_charbox(i)
        ch = textpage.get_text_range(i, 1)
        if not ch or ch.isspace():
            chars.append({"ch": ch, "x0": c[0], "y0": c[1], "x1": c[2], "y1": c[3], "h": c[3] - c[1]})
            continue
        x0, x1 = (c[0], c[2]) if c[0] <= c[2] else (c[2], c[0])
        y0, y1 = (c[1], c[3]) if c[1] <= c[3] else (c[3], c[1])
        if x1 < 0 or x0 > width or y1 < 0 or y0 > height:
            continue
        if invisible_boxes and _point_in_boxes((x0 + x1) / 2, (y0 + y1) / 2, invisible_boxes):
            continue
        chars.append({"ch": ch, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "h": y1 - y0})

    if not chars:
        return ""

    lines = _group_into_lines(chars)
    lines = _merge_orphan_punctuation_lines(lines)
    lines = _detect_columns_and_reorder(lines, page)
    lines = _infer_spaces(lines)
    lines = _strip_code_line_numbers(lines)
    lines = _classify_titles(lines)
    lines = _filter_chart_labels(lines)
    lines = _strip_arxiv_pagenum(lines)
    reconstructed = _lines_to_markdown(lines)

    # 质量回退检查 (detail.md §pdf 第10点)
    if _should_fallback_to_original(reconstructed, original_text, lines):
        return original_text
    return reconstructed


def _layout_garbled_line_fraction(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    garbled = 0
    for ln in lines:
        words = ln.split()
        if len(words) >= 6 and sum(1 for w in words if len(w) <= 2) / len(words) > 0.45:
            garbled += 1
    return garbled / len(lines)


def _plain_is_well_formed(plain: str) -> bool:
    plain = (plain or "").strip()
    if not plain:
        return False
    if re.search(r"\[\w+,\s", plain):
        return True
    if plain.count(" . . ") >= 2:
        return True
    words = re.findall(r"\S+", plain)
    if len(words) < 30:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    return avg_len >= 5.0


def _should_fallback_to_original(reconstructed: str, original: str, lines: list[dict]) -> bool:
    if not original or not original.strip():
        return False
    if not reconstructed or not reconstructed.strip():
        return True

    if _plain_is_well_formed(original):
        garbled_recon = _layout_garbled_line_fraction(reconstructed)
        garbled_plain = _layout_garbled_line_fraction(original)
        if garbled_recon >= 0.20 and garbled_plain < 0.08:
            return True

    recon_lines = [l for l in reconstructed.split("\n") if l.strip()]
    if not recon_lines:
        return True

    single_char_lines = sum(1 for l in recon_lines if len(l.strip()) == 1)
    single_char_ratio = single_char_lines / len(recon_lines)
    if single_char_ratio > QUALITY_SINGLE_CHAR_RATIO:
        return True

    punct_lines = sum(1 for l in recon_lines if all(c in "。，、；：？！.!?;:," for c in l.strip()))
    punct_ratio = punct_lines / len(recon_lines)
    if punct_ratio > QUALITY_PUNCT_RATIO:
        return True

    recon_words = reconstructed.split()
    if recon_words:
        single_char_words = sum(1 for w in recon_words if len(w) == 1 and w.isalpha())
        single_char_word_ratio = single_char_words / len(recon_words)
        if single_char_word_ratio > 0.20:
            return True

        recon_avg_len = sum(len(w) for w in recon_words) / len(recon_words)
        if recon_avg_len < 2.5:
            return True

    short_mixed_words = sum(
        1 for w in recon_words
        if 2 <= len(w) <= 3 and any(c.isupper() for c in w) and any(c.islower() for c in w)
    )
    if recon_words and short_mixed_words / len(recon_words) > 0.15:
        return True

    original_words = original.split()
    if original_words and recon_words:
        orig_avg_len = sum(len(w) for w in original_words) / len(original_words)
        recon_avg_len = sum(len(w) for w in recon_words) / len(recon_words)
        if orig_avg_len > 3 and recon_avg_len < orig_avg_len * 0.6:
            return True

    # 标题存活验证：原文第一行有效内容应在重建文本中存活
    for ln in original.splitlines():
        probe = ln.strip()
        if len(probe) < 24:
            continue
        alnum = "".join(c for c in probe if c.isalnum())[:16]
        if len(alnum) < 12:
            continue
        layout_alnum = "".join(c for c in reconstructed if c.isalnum())
        if alnum not in layout_alnum:
            return True
        break

    return False


def _group_into_lines(chars: list[dict]) -> list[dict]:
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (-c["y1"], c["x0"]))
    lines: list[dict] = []
    current_line_chars: list[dict] = []
    current_y = None
    tolerance = 3.0

    for ch in sorted_chars:
        if current_y is None or abs(ch["y1"] - current_y) <= tolerance:
            current_line_chars.append(ch)
            if current_y is None:
                current_y = ch["y1"]
        else:
            if current_line_chars:
                lines.append(_finalize_line(current_line_chars))
            current_line_chars = [ch]
            current_y = ch["y1"]

    if current_line_chars:
        lines.append(_finalize_line(current_line_chars))
    return lines


def _finalize_line(chars: list[dict]) -> dict:
    chars_sorted = sorted(chars, key=lambda c: c["x0"])
    text = "".join(c["ch"] for c in chars_sorted)
    heights = [c["h"] for c in chars_sorted if c["h"] > 0]
    avg_h = median(heights) if heights else 10.0
    widths = [c["x1"] - c["x0"] for c in chars_sorted if c["x1"] - c["x0"] > 0]
    median_w = median(widths) if widths else 5.0
    return {
        "text": text,
        "height": avg_h,
        "median_char_w": median_w,
        "chars": chars_sorted,
        "is_title": False,
        "title_level": 0,
    }


def _merge_orphan_punctuation_lines(lines: list[dict]) -> list[dict]:
    if not lines:
        return []
    merged: list[dict] = []
    for ln in lines:
        t = ln["text"].strip()
        if (
            merged
            and t
            and len(t) <= 4
            and all(c in ".,;:!?…·" or c.isspace() for c in t)
        ):
            suffix = "".join(t.split())
            prev = merged[-1]["text"]
            if suffix and prev and not prev.endswith((" ", "-")):
                merged[-1]["text"] = prev + suffix
            else:
                merged[-1]["text"] = (prev + " " + t).strip()
            continue
        merged.append(dict(ln))
    return merged


def _detect_columns_and_reorder(lines: list[dict], page: pdfium.PdfPage) -> list[dict]:
    all_chars = []
    for ln in lines:
        all_chars.extend(ln.get("chars", []))
    if len(all_chars) < 20:
        return lines
    width, height = page.get_size()
    scale = 2.0
    columns = _filter_reading_columns(all_chars, scale, width)
    if len(columns) <= 1:
        return lines
    reordered: list[dict] = []
    for col_chars in columns:
        col_lines = _group_into_lines(col_chars)
        reordered.extend(col_lines)
    return reordered


def _find_split(chars: list[dict], min_gap: float) -> float | None:
    intervals = sorted(((c["x0"], c["x1"]) for c in chars), key=lambda iv: iv[0])
    if not intervals:
        return None
    cur_end = intervals[0][1]
    best_gap, best_cut = 0.0, None
    for a, b in intervals[1:]:
        gap = a - cur_end
        if gap >= min_gap and gap > best_gap:
            best_gap, best_cut = gap, cur_end + gap / 2
        if b > cur_end:
            cur_end = b
    return best_cut


def _split_columns(chars: list[dict], scale: float, width: float, depth: int = 0) -> list[list[dict]]:
    if len(chars) <= 1 or depth > 10:
        return [chars]
    min_gap = max(scale * 2.5, width * 0.04)
    cut = _find_split(chars, min_gap)
    if cut is None:
        return [chars]
    left = [c for c in chars if (c["x0"] + c["x1"]) / 2 < cut]
    right = [c for c in chars if (c["x0"] + c["x1"]) / 2 >= cut]
    if not left or not right:
        return [chars]
    return _split_columns(left, scale, width, depth + 1) + _split_columns(
        right, scale, width, depth + 1
    )


def _column_x_span(chars: list[dict]) -> float:
    if not chars:
        return 0.0
    return max(c["x1"] for c in chars) - min(c["x0"] for c in chars)


def _is_artifact_column(chars: list[dict], width: float) -> bool:
    if not chars or width <= 0:
        return True
    span = _column_x_span(chars)
    if span <= 0:
        return True
    lines = _group_into_lines(chars)
    single_frac = sum(1 for ln in lines if len(ln["text"]) <= 2) / len(lines) if lines else 0.0
    narrow = span / width < NOISE_COL_WIDTH_RATIO
    if narrow and single_frac >= 0.45:
        return True
    ys = [(c["y0"] + c["y1"]) / 2 for c in chars]
    y_span = max(ys) - min(ys)
    if y_span > span * 3.5 and len(chars) >= 8 and single_frac >= 0.35:
        return True
    return False


def _filter_reading_columns(chars: list[dict], scale: float, width: float) -> list[list[dict]]:
    cols = _split_columns(chars, scale, width)
    kept = [c for c in cols if not _is_artifact_column(c, width)]
    if kept:
        return kept
    if len(cols) > 1:
        return [max(cols, key=_column_x_span)]
    return cols


def _infer_spaces(lines: list[dict]) -> list[dict]:
    for line in lines:
        chars = line.get("chars", [])
        if len(chars) < 2:
            continue
        median_w = line["median_char_w"]
        new_text_parts = [chars[0]["ch"]]
        for i in range(1, len(chars)):
            gap = chars[i]["x0"] - chars[i - 1]["x1"]
            if gap > median_w * SPACE_GAP_RATIO:
                new_text_parts.append(" ")
            new_text_parts.append(chars[i]["ch"])
        line["text"] = "".join(new_text_parts)
    return lines


def _strip_code_line_numbers(lines: list[dict]) -> list[dict]:
    """Remove runs of consecutive number-only lines (code editor line numbers)."""
    result: list[dict] = []
    i = 0
    while i < len(lines):
        t = lines[i]["text"].strip()
        if re.match(r"^\d{1,3}$", t):
            run_start = i
            nums = [int(t)]
            j = i + 1
            while j < len(lines):
                tj = lines[j]["text"].strip()
                if re.match(r"^\d{1,3}$", tj):
                    nums.append(int(tj))
                    j += 1
                else:
                    break
            run_len = j - run_start
            if run_len >= LINENUM_RUN_MIN and _is_linenum_sequence(nums):
                i = j
                continue
        result.append(lines[i])
        i += 1
    return result


def _is_linenum_sequence(nums: list[int]) -> bool:
    """Check if numbers form line-number patterns (consecutive or restarting from 1)."""
    if not nums:
        return False
    consecutive = 0
    restarts = 0
    for k in range(1, len(nums)):
        if nums[k] == nums[k - 1] + 1:
            consecutive += 1
        elif nums[k] == 1 and nums[k - 1] > 1:
            restarts += 1
            consecutive += 1
    return consecutive >= len(nums) * 0.6 or restarts >= 1


def _classify_titles(lines: list[dict]) -> list[dict]:
    if not lines:
        return lines
    heights = [ln["height"] for ln in lines if ln["height"] > 0]
    if not heights:
        return lines
    base_h = median(heights)
    title_count = 0
    for ln in lines:
        text = ln["text"].strip()
        ratio = ln["height"] / base_h if base_h > 0 else 0
        is_candidate = (
            len(text) >= TITLE_MIN_LEN
            and len(text) <= TITLE_MAX_LEN
            and (not text or text[-1] not in TITLE_END_PUNCT)
        )
        # Pattern-based Chinese heading detection
        if _CN_H1_RE.match(text) and len(text) <= TITLE_MAX_LEN:
            ln["is_title"] = True
            ln["title_level"] = 1
            title_count += 1
            continue
        if _CN_H2_RE.match(text) and len(text) <= TITLE_MAX_LEN:
            ln["is_title"] = True
            ln["title_level"] = 2
            title_count += 1
            continue
        if _CN_H3_RE.match(text) and len(text) <= TITLE_MAX_LEN:
            ln["is_title"] = True
            ln["title_level"] = 3
            title_count += 1
            continue
        # Font-size based detection
        if ratio >= TITLE_H1_RATIO and is_candidate:
            ln["is_title"] = True
            ln["title_level"] = 1
            title_count += 1
        elif ratio >= TITLE_H2_RATIO and is_candidate:
            ln["is_title"] = True
            ln["title_level"] = 2
            title_count += 1
        elif ratio >= TITLE_H3_RATIO and is_candidate:
            ln["is_title"] = True
            ln["title_level"] = 3
            title_count += 1

    if len(lines) > 0 and title_count / len(lines) > TITLE_NOISE_THRESHOLD:
        for ln in lines:
            ln["is_title"] = False
            ln["title_level"] = 0
    return lines


def _filter_chart_labels(lines: list[dict]) -> list[dict]:
    result = []
    i = 0
    while i < len(lines):
        if _is_chart_debris_line(lines[i]["text"]):
            run_start = i
            while i < len(lines) and (_is_chart_debris_line(lines[i]["text"]) or not lines[i]["text"].strip()):
                i += 1
            if i - run_start >= 3:
                continue
            else:
                result.extend(lines[run_start:i])
        else:
            result.append(lines[i])
            i += 1
    return result


def _is_chart_debris_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _CHART_DEBRIS_LINE_RE.match(t):
        return True
    if _CHART_LAYER_RE.match(t):
        return True
    if re.fullmatch(r"[\d\s.()\-]+", t) and len(t) <= 24 and sum(c.isdigit() for c in t) >= 3:
        return True
    return False


def _strip_arxiv_pagenum(lines: list[dict]) -> list[dict]:
    result = []
    for ln in lines:
        t = ln["text"].strip()
        if t.startswith("arXiv:"):
            continue
        if re.match(r"^\d{1,3}$", t):
            continue
        result.append(ln)
    return result


def _detect_code_lines(lines: list[dict]) -> list[dict]:
    """Mark lines that look like source code."""
    for ln in lines:
        if ln.get("is_title"):
            ln["is_code"] = False
            continue
        text = ln["text"]
        ln["is_code"] = bool(_CODE_LINE_RE.match(text))
    return lines


def _lines_to_markdown(lines: list[dict]) -> str:
    parts: list[str] = []
    for ln in lines:
        text = ln["text"].strip()
        if not text:
            parts.append("")
            continue
        if ln["is_title"]:
            prefix = "#" * ln["title_level"]
            parts.append(f"{prefix} {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


def _remove_headers_footers(pages_text: list[str]) -> list[str]:
    if len(pages_text) < HEADER_FOOTER_MIN_PAGES:
        return pages_text

    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    text_page_count = sum(1 for t in pages_text if t.strip())

    for text in pages_text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue
        fl = lines[0]
        ll = lines[-1]
        if len(fl) < 80:
            first_lines[fl] = first_lines.get(fl, 0) + 1
        if len(ll) < 80:
            last_lines[ll] = last_lines.get(ll, 0) + 1

    threshold = max(HEADER_FOOTER_MIN_PAGES, int(text_page_count * HEADER_FOOTER_PAGE_RATIO))
    hf_set = set()
    for line, count in {**first_lines, **last_lines}.items():
        if count >= threshold:
            hf_set.add(line)

    if not hf_set:
        return pages_text

    result = []
    for text in pages_text:
        lines = text.split("\n")
        filtered = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped in hf_set and (i == 0 or i == len(lines) - 1):
                continue
            filtered.append(line)
        result.append("\n".join(filtered))
    return result


def _collect_embedded_candidates(
    page: pdfium.PdfPage, page_idx: int
) -> list[dict]:
    """Phase 1: collect raw image candidates from a text page (no encoding yet)."""
    import hashlib

    width, height = page.get_size()
    page_area = width * height
    if page_area == 0:
        return []

    candidates: list[dict] = []
    for obj in page.get_objects(filter=(3,)):
        m = obj.get_matrix()
        img_w = abs(m.a)
        img_h = abs(m.d)

        if img_w < EMBED_IMG_MIN_PX or img_h < EMBED_IMG_MIN_PX:
            continue
        area_ratio = (img_w * img_h) / page_area
        if area_ratio < EMBED_IMG_AREA_RATIO:
            continue

        try:
            bitmap = obj.get_bitmap()
            pil_image = bitmap.to_pil()
        except Exception:
            continue

        content_hash = hashlib.md5(pil_image.tobytes()).hexdigest()
        # y_top: PDF 坐标系原点在左下，matrix.f 是图片底边 y
        y_top = m.f + img_h

        candidates.append({
            "page": page_idx,
            "pil": pil_image,
            "width": pil_image.width,
            "height": pil_image.height,
            "area_ratio": area_ratio,
            "hash": content_hash,
            "y_top": y_top,
        })
    return candidates


def _filter_embedded_candidates(
    all_candidates: list[dict], num_text_pages: int
) -> list[dict]:
    """Phase 2: cross-page dedup (logo/watermark), in-page dedup, count cap."""
    from collections import defaultdict

    # 跨页重复检测：同一 hash 出现在 ≥50% 文字页 → logo/水印
    hash_pages: dict[str, set] = defaultdict(set)
    for c in all_candidates:
        hash_pages[c["hash"]].add(c["page"])
    repeat_threshold = max(2, int(num_text_pages * EMBED_REPEAT_PAGE_FRAC))
    banned_hashes = {h for h, pages in hash_pages.items() if len(pages) >= repeat_threshold}

    kept: list[dict] = []
    seen_page_hash: set[tuple] = set()
    for c in all_candidates:
        if c["hash"] in banned_hashes:
            continue
        # 页内精确去重
        key = (c["page"], c["hash"])
        if key in seen_page_hash:
            continue
        seen_page_hash.add(key)
        kept.append(c)
        if len(kept) >= EMBED_IMG_MAX_COUNT:
            break

    # 按页码 + 位置排序（阅读顺序）
    kept.sort(key=lambda c: (c["page"], -c["y_top"]))
    return kept


def _encode_embedded_image(pil_image) -> bytes | None:
    """Phase 3: resize large images + JPEG compress."""
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    max_edge = max(pil_image.size)
    if max_edge > EMBED_IMG_MAX_EDGE:
        ratio = EMBED_IMG_MAX_EDGE / max_edge
        new_size = (max(1, int(pil_image.width * ratio)), max(1, int(pil_image.height * ratio)))
        pil_image = pil_image.resize(new_size)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85, optimize=True)
    data = buf.getvalue()
    return data if len(data) >= 500 else None


def _detect_vector_figures(page: pdfium.PdfPage, page_idx: int) -> list[dict]:
    width, height = page.get_size()
    page_area = width * height
    if page_area == 0:
        return []

    path_count = 0
    for _ in page.get_objects(filter=(2,)):
        path_count += 1
    if path_count < VECTOR_FIG_MIN_PATHS:
        return []

    has_bitmap = False
    for obj in page.get_objects(filter=(3,)):
        m = obj.get_matrix()
        img_w = abs(m.a)
        img_h = abs(m.d)
        if img_w >= EMBED_IMG_MIN_PX and img_h >= EMBED_IMG_MIN_PX:
            if (img_w * img_h) / page_area >= EMBED_IMG_AREA_RATIO:
                has_bitmap = True
                break
    if has_bitmap:
        return []

    caption_y = _find_figure_caption_y(page, height)
    scale = _effective_scale(page)
    bitmap = page.render(scale=scale)
    pil_full = bitmap.to_pil().convert("RGB")
    pw, ph = pil_full.size

    if caption_y is not None:
        top = max(0, int(ph * 0.08))
        bottom = min(ph, int((caption_y / height) * ph))
        left = int(pw * 0.08)
        right = int(pw * 0.92)
        if bottom - top < ph * 0.1:
            top = max(0, bottom - int(ph * 0.35))
    else:
        left = int(pw * 0.1)
        right = int(pw * 0.9)
        top = int(ph * 0.1)
        bottom = int(ph * 0.9)

    cropped = pil_full.crop((left, top, right, bottom))
    cw, ch = cropped.size
    crop_area_ratio = (cw * ch) / (pw * ph)

    if crop_area_ratio < VECTOR_FIG_MIN_AREA or crop_area_ratio > VECTOR_FIG_MAX_AREA:
        return []

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85, optimize=True)
    data = buf.getvalue()
    if len(data) < 1000:
        return []

    y_top = height - (top / ph * height) if caption_y else height * 0.5
    return [{"page": page_idx, "data": data, "y_top": y_top}]


def _find_figure_caption_y(page: pdfium.PdfPage, page_height: float) -> float | None:
    textpage = page.get_textpage()
    n = textpage.count_chars()
    if n == 0:
        return None
    buf: list[str] = []
    last_y = None
    for i in range(n):
        ch = textpage.get_text_range(i, 1)
        c = textpage.get_charbox(i)
        y = (c[1] + c[3]) / 2
        if last_y is not None and abs(y - last_y) > 3.0:
            line_text = "".join(buf).strip()
            if _FIGURE_CAPTION_RE.match(line_text):
                return last_y
            buf = []
        buf.append(ch)
        last_y = y
    if buf:
        line_text = "".join(buf).strip()
        if _FIGURE_CAPTION_RE.match(line_text):
            return last_y
    return None


@register("pdf")
def parse_pdf(data: bytes, filename: str, **kwargs) -> ParseResult:
    with _PDFIUM_LOCK:
        return _parse_pdf_locked(data, filename, **kwargs)


def _parse_pdf_locked(data: bytes, filename: str, **kwargs) -> ParseResult:
    force_scanned = kwargs.get("force_scanned", FORCE_SCANNED_PDF)
    doc = pdfium.PdfDocument(data)
    images: list[ImageRef] = []
    pages_text: list[str] = []
    page_is_scan: list[bool] = []
    all_embed_candidates: list[dict] = []
    text_page_count = 0
    scanned_page_count = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        is_scan = force_scanned or _is_scan_page(page)
        if is_scan:
            page_is_scan.append(True)
            scanned_page_count += 1
            img_data = _render_page_image(page)
            ref = ImageRef(data=img_data, mime="image/jpeg")
            images.append(ref)
            pages_text.append(f"![scan_page_{page_idx}](images/{ref.id}.jpg)")
        else:
            page_is_scan.append(False)
            text = _extract_text_page(page)
            if _is_formula_page(page, text):
                page_is_scan[-1] = True
                scanned_page_count += 1
                img_data = _render_page_image(page)
                ref = ImageRef(data=img_data, mime="image/jpeg")
                images.append(ref)
                pages_text.append(f"![formula_page_{page_idx}](images/{ref.id}.jpg)")
                continue
            text_page_count += 1
            all_embed_candidates.extend(_collect_embedded_candidates(page, page_idx))
            vec_figs = _detect_vector_figures(page, page_idx)
            for vf in vec_figs:
                ref = ImageRef(data=vf["data"], mime="image/jpeg")
                images.append(ref)
                placeholder = f"![vec_fig_{page_idx}](images/{ref.id}.jpg)"
                text = text + "\n\n" + placeholder if text.strip() else placeholder
            pages_text.append(text)

    kept = _filter_embedded_candidates(all_embed_candidates, text_page_count)

    page_placeholders: dict[int, list[str]] = {}
    for c in kept:
        img_data = _encode_embedded_image(c["pil"])
        if not img_data:
            continue
        ref = ImageRef(data=img_data, mime="image/jpeg")
        images.append(ref)
        placeholder = f"![fig_{c['page']}](images/{ref.id}.jpg)"
        page_placeholders.setdefault(c["page"], []).append(placeholder)

    for pg_idx, phs in page_placeholders.items():
        if pg_idx < len(pages_text):
            existing = pages_text[pg_idx]
            pages_text[pg_idx] = existing + "\n\n" + "\n\n".join(phs) if existing.strip() else "\n\n".join(phs)

    pages_text = _remove_headers_footers(pages_text)
    markdown = "\n\n".join(pages_text)
    markdown = _postprocess_markdown(markdown)
    page_count = len(doc)
    doc.close()

    metadata = {
        "page_count": page_count,
        "scanned_page_count": scanned_page_count,
        "text_page_count": text_page_count,
        "embedded_image_count": len(kept),
    }
    return ParseResult(markdown=markdown, images=images, metadata=metadata)


def _sanitize_pdf_text(text: str) -> str:
    if not text:
        return text
    text = _PDF_ARTIFACT_RE.sub("", text)
    text = _PDF_ARTIFACT_JOIN_RE.sub(r"\1\2", text)
    return text


def _is_body_paragraph_line(text: str) -> bool:
    t = text.strip()
    if len(t) < 48:
        return False
    return len(t.split()) >= 8


def _is_figure_interior_line(text: str) -> bool:
    t = text.strip()
    if not t or _FIGURE_CAPTION_RE.match(t):
        return False
    if re.match(r"^\d{1,3}$", t):
        return True
    if _is_body_paragraph_line(t):
        return False
    if _is_chart_debris_line(t):
        return True
    if t.endswith((".", "。", "!", "?", "！")) and len(t) >= 15:
        return False
    if len(t.split()) >= 7:
        return False
    if len(t) <= 40:
        return True
    return False


def _strip_lines_above_figure_captions(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        if _FIGURE_CAPTION_RE.match(ln.strip()):
            while out and _is_figure_interior_line(out[-1]):
                out.pop()
            out.append(ln)
        else:
            out.append(ln)
    return "\n".join(out)


def _postprocess_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _sanitize_pdf_text(text)
    text = _strip_linenum_runs_text(text)
    text = _strip_lines_above_figure_captions(text)
    text = _inject_headings(text)
    text = _wrap_code_blocks(text)
    text = _merge_paragraph_lines(text)
    return text


def _strip_linenum_runs_text(text: str) -> str:
    """Remove runs of 3+ consecutive number-only lines from text."""
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if re.match(r"^\d{1,3}$", stripped):
            j = i
            nums: list[int] = []
            while j < len(lines) and re.match(r"^\d{1,3}$", lines[j].strip()):
                nums.append(int(lines[j].strip()))
                j += 1
            run_len = j - i
            if run_len >= LINENUM_RUN_MIN and _is_linenum_sequence(nums):
                i = j
                continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


_MD_HEADING_RE = re.compile(r"^#{1,3}\s")
_REAL_HEADING_RE = re.compile(
    r"^#{1,3}\s+(?:[一二三四五六七八九十]+、|\d+\.\d+\s|步骤\d+|Q\d+|第\d+页摘要)"
)


def _inject_headings(text: str) -> str:
    """Add markdown heading markers to lines matching Chinese heading patterns.
    Skips lines inside code fences."""
    lines = text.split("\n")
    result: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue
        if _MD_HEADING_RE.match(stripped):
            result.append(line)
            continue
        if _CN_H1_RE.match(stripped) and len(stripped) <= TITLE_MAX_LEN:
            result.append(f"# {stripped}")
        elif _CN_H2_RE.match(stripped) and len(stripped) <= TITLE_MAX_LEN:
            result.append(f"## {stripped}")
        elif _CN_H3_RE.match(stripped) and len(stripped) <= TITLE_MAX_LEN:
            result.append(f"### {stripped}")
        else:
            result.append(line)
    return "\n".join(result)


def _merge_paragraph_lines(text: str) -> str:
    """Merge visual line breaks within paragraphs (runs after code fencing)."""
    lines = text.split("\n")
    result: list[str] = []
    in_fence = False
    buf: list[str] = []

    def flush():
        if buf:
            result.append("".join(buf))
            buf.clear()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue
        if not stripped:
            flush()
            result.append("")
            continue
        if stripped.startswith("#"):
            flush()
            result.append(line)
            continue
        if _LIST_ITEM_RE.match(stripped):
            flush()
            result.append(line)
            continue
        if _CODE_LINE_RE.match(stripped):
            flush()
            result.append(line)
            continue

        buf.append(stripped)

        if _CN_PARA_BREAK_RE.search(stripped):
            flush()
            continue

        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (
            not nxt
            or nxt.startswith("#")
            or nxt.startswith("```")
            or _LIST_ITEM_RE.match(nxt)
            or _CODE_LINE_RE.match(nxt)
        ):
            flush()

    flush()
    return "\n".join(result)


def _wrap_code_blocks(text: str) -> str:
    """Wrap consecutive code-like lines in markdown code fences.
    First strips any existing code fence markers from the source text."""
    lines = text.split("\n")
    # Strip existing code fence markers from source
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    lines = cleaned

    result: list[str] = []
    in_code = False
    blank_count = 0
    for line in lines:
        stripped = line.strip()
        # Only Chinese-pattern headings break code blocks, not # comments
        if _REAL_HEADING_RE.match(stripped):
            if in_code:
                result.append("```")
                in_code = False
            blank_count = 0
            result.append(line)
            continue
        if not stripped:
            if in_code:
                blank_count += 1
                if blank_count > 1:
                    result.append("```")
                    in_code = False
                    blank_count = 0
                else:
                    result.append(line)
            else:
                result.append(line)
            continue
        is_code_start = bool(_CODE_LINE_RE.match(stripped))
        is_code_cont = bool(_CODE_CONT_RE.match(stripped))
        # Only close code block for lines with Chinese text and no code features
        has_chinese = bool(_HAS_CHINESE_RE.search(stripped))
        has_code_feat = bool(_HAS_CODE_FEATURE_RE.search(stripped))
        is_prose = has_chinese and not has_code_feat and not is_code_start and not is_code_cont
        if in_code:
            if is_prose:
                result.append("```")
                in_code = False
                blank_count = 0
                result.append(line)
            else:
                blank_count = 0
                result.append(line)
        else:
            if is_code_start:
                result.append("```python")
                in_code = True
                blank_count = 0
                result.append(line)
            else:
                result.append(line)
    if in_code:
        result.append("```")
    return "\n".join(result)
