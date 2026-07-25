from __future__ import annotations

import io
import re
from statistics import median

import pypdfium2 as pdfium

from .base import ImageRef, ParseResult
from .registry import register

# detail.md §pdf
SCAN_IMAGE_AREA_RATIO = 0.5
SCAN_MIN_CHARS = 10
SCAN_IMAGE_AREA_RATIO_LOW = 0.1
TITLE_H1_RATIO = 2.0
TITLE_H2_RATIO = 1.6
TITLE_H3_RATIO = 1.35
TITLE_NOISE_THRESHOLD = 0.4
TITLE_MIN_LEN = 2
TITLE_MAX_LEN = 80
HEADER_FOOTER_PAGE_RATIO = 0.6
HEADER_FOOTER_MIN_PAGES = 4
EMBED_IMG_MIN_PX = 80
EMBED_IMG_AREA_RATIO = 0.01
EMBED_IMG_MAX_COUNT = 50
QUALITY_SINGLE_CHAR_RATIO = 0.18
QUALITY_PUNCT_RATIO = 0.12
QUALITY_DISORDER_RATIO = 0.20
QUALITY_ORIGINAL_DISORDER = 0.08
NOISE_COL_WIDTH_RATIO = 0.12
SPACE_GAP_RATIO = 0.4
TITLE_END_PUNCT = set("。？！，；：.?!;,")
LINENUM_RUN_MIN = 3

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
_HAS_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_HAS_CODE_FEATURE_RE = re.compile(r"[=\(\)\[\]{}]|\w+\.\w+|self\.|:\s*$|,\s*$")


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


def _render_page_image(page: pdfium.PdfPage, scale: float = 2.0) -> bytes:
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _extract_text_page(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    char_count = textpage.count_chars()
    if char_count == 0:
        return ""

    # 先获取原始文本作为回退选项
    original_text = textpage.get_text_range(0, char_count)

    chars: list[dict] = []
    for i in range(char_count):
        c = textpage.get_charbox(i)
        ch = textpage.get_text_range(i, 1)
        if not ch or ch.isspace():
            chars.append({"ch": ch, "x0": c[0], "y0": c[1], "x1": c[2], "y1": c[3], "h": c[3] - c[1]})
            continue
        chars.append({"ch": ch, "x0": c[0], "y0": c[1], "x1": c[2], "y1": c[3], "h": c[3] - c[1]})

    if not chars:
        return ""

    lines = _group_into_lines(chars)
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


def _should_fallback_to_original(reconstructed: str, original: str, lines: list[dict]) -> bool:
    """检查重建文本质量，决定是否回退到原始文本"""
    if not original or not original.strip():
        return False
    if not reconstructed or not reconstructed.strip():
        return True

    recon_lines = [l for l in reconstructed.split("\n") if l.strip()]
    if not recon_lines:
        return True

    # 检查单字符行比例
    single_char_lines = sum(1 for l in recon_lines if len(l.strip()) == 1)
    single_char_ratio = single_char_lines / len(recon_lines)
    if single_char_ratio > QUALITY_SINGLE_CHAR_RATIO:
        return True

    # 检查纯标点行比例
    punct_lines = sum(1 for l in recon_lines if all(c in "。，、；：？！.!?;:," for c in l.strip()))
    punct_ratio = punct_lines / len(recon_lines)
    if punct_ratio > QUALITY_PUNCT_RATIO:
        return True

    # 检查“碎词”模式：英文单词被拆成单字符（如 "d i d l" 或 "wor p ece"）
    recon_words = reconstructed.split()
    if recon_words:
        single_char_words = sum(1 for w in recon_words if len(w) == 1 and w.isalpha())
        single_char_word_ratio = single_char_words / len(recon_words)
        # 如果超过20%的词是单字符，说明空格推断有问题
        if single_char_word_ratio > 0.20:
            return True

        # 检查平均词长：正常英文/代码平均词长>3，如果重建后平均词长<2.5则异常
        recon_avg_len = sum(len(w) for w in recon_words) / len(recon_words)
        if recon_avg_len < 2.5:
            return True

    # 检查代码被拆散的模式（如 "Posi ti onalEncodi ng" 或 "i ni t"）
    # 特征：大量 2-3 字符的“词”且包含大小写混合
    short_mixed_words = sum(
        1 for w in recon_words
        if 2 <= len(w) <= 3 and any(c.isupper() for c in w) and any(c.islower() for c in w)
    )
    if recon_words and short_mixed_words / len(recon_words) > 0.15:
        return True

    # 比较原文和重建文本的平均词长
    original_words = original.split()
    if original_words and recon_words:
        orig_avg_len = sum(len(w) for w in original_words) / len(original_words)
        recon_avg_len = sum(len(w) for w in recon_words) / len(recon_words)
        # 如果原文平均词长明显更长（>1.5倍），说明重建质量差
        if orig_avg_len > 3 and recon_avg_len < orig_avg_len * 0.6:
            return True

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


def _detect_columns_and_reorder(lines: list[dict], page: pdfium.PdfPage) -> list[dict]:
    return lines


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
        if _is_numeric_short(lines[i]["text"]):
            run_start = i
            while i < len(lines) and _is_numeric_short(lines[i]["text"]):
                i += 1
            if i - run_start >= 3:
                continue
            else:
                result.extend(lines[run_start:i])
        else:
            result.append(lines[i])
            i += 1
    return result


def _is_numeric_short(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 20:
        return False
    return bool(re.match(r"^[\d\s\.\-\+\(\)eE,]+$", t))


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


@register("pdf")
def parse_pdf(data: bytes, filename: str) -> ParseResult:
    doc = pdfium.PdfDocument(data)
    images: list[ImageRef] = []
    pages_text: list[str] = []
    page_is_scan: list[bool] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        if _is_scan_page(page):
            page_is_scan.append(True)
            img_data = _render_page_image(page)
            ref = ImageRef(data=img_data, mime="image/jpeg")
            images.append(ref)
            pages_text.append(f"![scan_page_{page_idx}](images/{ref.id}.jpg)")
        else:
            page_is_scan.append(False)
            text = _extract_text_page(page)
            pages_text.append(text)

    pages_text = _remove_headers_footers(pages_text)
    markdown = "\n\n".join(pages_text)
    markdown = _postprocess_markdown(markdown)
    doc.close()
    return ParseResult(markdown=markdown, images=images)


def _postprocess_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_linenum_runs_text(text)
    text = _inject_headings(text)
    text = _wrap_code_blocks(text)
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
