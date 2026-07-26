from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from ..llm.client import get_llm_client
from ..parsing.base import ImageRef
from ..settings import settings

logger = logging.getLogger(__name__)

VLM_CACHE_DIR: Path = settings.object_storage_root / "cache" / "vlm"
VLM_RETRY_ATTEMPTS = 2


def _cache_path(image: ImageRef) -> Path:
    return VLM_CACHE_DIR / f"{hashlib.sha256(image.data).hexdigest()}.json"

VLM_OCR_PROMPT = (
    "You are an OCR assistant. Extract all body text content from this document image "
    "and output in pure Markdown format. Ignore headers/footers. Use Markdown tables for tables. "
    "Use LaTeX ($/$$) for formulas. Maintain original reading order. "
    "Only output extracted content. If no text, reply 'No text content'."
)

VLM_CAPTION_PROMPT = (
    "Provide a brief and concise description of the main content of the image in Chinese."
)


async def process_image_cascade(
    image: ImageRef,
    use_ocr: bool = True,
    use_vlm: bool = True,
) -> dict:
    cache_file = _cache_path(image)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    client = get_llm_client()
    result = {"ocr_text": "", "caption": ""}

    vlm_retry = retry(
        stop=stop_after_attempt(VLM_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )

    if use_ocr:
        try:
            ocr_text = await vlm_retry(client.vlm)(image.base64, VLM_OCR_PROMPT, mime=image.mime, purpose="vlm:ocr")
            if ocr_text and ocr_text.strip() != "No text content":
                result["ocr_text"] = ocr_text.strip()
        except Exception as e:
            logger.warning("[VLM] OCR 失败 image=%s error=%r", image.id, e)

    if use_vlm and not result["ocr_text"]:
        try:
            caption = await vlm_retry(client.vlm)(image.base64, VLM_CAPTION_PROMPT, mime=image.mime, purpose="vlm:caption")
            if caption:
                result["caption"] = caption.strip()
        except Exception as e:
            logger.warning("[VLM] caption 失败 image=%s error=%r", image.id, e)

    if result["ocr_text"] or result["caption"]:
        VLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


async def enrich_markdown_with_images(markdown: str, images: list[ImageRef]) -> str:
    """Run VLM OCR on images and replace ![alt](images/{id}.*) placeholders in-place."""
    if not images:
        return markdown

    logger.info("[VLM] 开始图片提取: images=%d", len(images))
    sem = asyncio.Semaphore(settings.vlm_concurrency)

    async def _process(i: int, img: ImageRef) -> tuple[int, str]:
        async with sem:
            try:
                res = await process_image_cascade(img)
                content = res["ocr_text"] or res["caption"]
                if content:
                    logger.info("[VLM] 图片 %d/%d 提取成功: %d chars", i + 1, len(images), len(content))
                    return (i, content)
            except Exception as e:
                logger.warning("[VLM] 图片 %d/%d 处理失败: %r", i + 1, len(images), e)
            return (i, "")

    results = await asyncio.gather(*[_process(i, img) for i, img in enumerate(images)])

    # 原地替换占位符：alt 文本和扩展名各解析器不同，仅按 ref.id 匹配
    enriched = markdown
    for i, content in results:
        if not content:
            continue
        ref = images[i]
        pattern = re.compile(rf"!\[[^\]]*\]\(images/{re.escape(ref.id)}[^)]*\)")
        enriched, n = pattern.subn(lambda _m: content, enriched, count=1)
        if n == 0:
            enriched += "\n\n" + content

    logger.info("[VLM] 完成: %d -> %d chars", len(markdown), len(enriched))
    return enriched
