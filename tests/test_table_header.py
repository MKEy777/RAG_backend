from __future__ import annotations

from knowlink.chunking.table_header import annotate_table_headers


def test_header_injected_into_continuation():
    header = "| 姓名 | 部门 | 职级 |\n| --- | --- | --- |"
    rows_a = "\n".join(f"| 员工{i} | 研发 | P{i} |" for i in range(1, 6))
    rows_b = "\n".join(f"| 员工{i} | 市场 | P{i} |" for i in range(6, 11))
    out = annotate_table_headers([f"{header}\n{rows_a}", rows_b])
    assert out[0] == f"{header}\n{rows_a}"
    assert out[1].startswith("| 姓名 | 部门 | 职级 |")
    assert rows_b in out[1]


def test_non_table_chunks_unchanged():
    chunks = ["普通段落一。", "普通段落二。"]
    assert annotate_table_headers(chunks) == chunks


def test_column_mismatch_stops_injection():
    header = "| A | B | C |\n| --- | --- | --- |"
    out = annotate_table_headers([f"{header}\n| 1 | 2 | 3 |", "| 两列 | 数据 |"])
    assert out[1] == "| 两列 | 数据 |"


def test_table_closed_by_prose_no_injection():
    header = "| A | B |\n| --- | --- |"
    out = annotate_table_headers([f"{header}\n| 1 | 2 |\n后面是普通文字收尾", "| 3 | 4 |"])
    assert out[1] == "| 3 | 4 |"
