"""Quick parse+chunk test for the Transformer PDF."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.parsing.registry import parse_file
from pyrag.chunking import chunk_document
from pyrag.chunking.parent_child import ParentChildPair

PDF_PATH = Path(r"c:\Users\VECTOR\Desktop\RAG\测试文件") / "【论文精读 & 面试题】Transformer论文逐段精读.pdf"


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: {PDF_PATH} not found")
        return

    out_dir = Path(__file__).parent / "debug_output"
    out_dir.mkdir(exist_ok=True)

    # Parse
    data = PDF_PATH.read_bytes()
    result = parse_file(data, PDF_PATH.name)

    # Write markdown to file for inspection
    md_path = out_dir / "parsed_markdown.md"
    md_path.write_text(result.markdown, encoding="utf-8")
    print(f"Markdown written to {md_path} ({len(result.markdown)} chars, {len(result.images)} images)")

    # Chunk
    chunks = chunk_document(result.markdown, strategy="auto", enable_parent_child=True)

    # Write chunk report
    report_lines = []
    if chunks and isinstance(chunks[0], ParentChildPair):
        report_lines.append(f"Mode: parent-child")
        report_lines.append(f"Parent chunks: {len(chunks)}")
        total_children = sum(len(p.children) for p in chunks)
        report_lines.append(f"Child chunks: {total_children}")
        report_lines.append("")
        for i, pair in enumerate(chunks):
            report_lines.append(f"=== Parent {i} | title_path={pair.parent_title_path} | parent_len={len(pair.parent_text)} ===")
            for j, child in enumerate(pair.children):
                report_lines.append(f"  [Child {i}.{j}] len={len(child.text)}")
                report_lines.append(f"  {child.text[:300]}")
                report_lines.append("")
    else:
        report_lines.append(f"Mode: flat, chunks={len(chunks)}")
        for i, c in enumerate(chunks):
            report_lines.append(f"=== Chunk {i} | len={len(c.text)} ===")
            report_lines.append(c.text[:300])
            report_lines.append("")

    report_path = out_dir / "chunk_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Chunk report written to {report_path}")
    print(f"Parents: {len(chunks)}, Children: {sum(len(p.children) for p in chunks) if chunks and isinstance(chunks[0], ParentChildPair) else 'N/A'}")


if __name__ == "__main__":
    main()
