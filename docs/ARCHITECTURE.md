# pyrag 架构设计文档

> 本文档是项目架构的**唯一权威记录**。当架构发生变更时，必须同步更新本文件。
> 设计目标详见上级 `detail.md`（路径：`../detail.md`），参考实现 `../WeKnora/`。

## 一、项目目标

以纯 Python 复刻 `detail.md` 描述的「文档解析 + RAG + Wiki + 知识图谱」系统（无前端）。设计依据层级：

1. `detail.md`（设计目标最高依据，全部参数与流程出处）
2. 本文档（架构与里程碑）
3. `AGENTS.md`（编码规范与约束）

## 二、技术决策

### 2.1 已确认选型

- **存储（轻量版）**：SQLite（SQLAlchemy 2.0）+ 本地向量（numpy 暴力余弦，float16 落盘）+ bm25s 关键词索引 + Redis 仅作 Celery broker 与分布式锁
- **模型接入**：全部走 OpenAI 兼容 HTTP API
  - 嵌入：Qwen3-Embedding-4B
  - 对话 LLM（抽取、生成、综述）
  - 多模态 VLM（OCR / 图片描述）
  - 重排：BGE-Reranker（rerank / TEI 风格可配）
- **异步任务**：Celery + Redis（对应 WeKnora asynq）；Windows 开发用 `--pool=solo/threads`
- **对外接口**：FastAPI（上传/检索/SSE 问答/Wiki/图谱）
- **完整版预留**：通过 `storage/interfaces.py` 抽象接口，保留 MySQL+Elasticsearch+Redis(全功能) 切换点；`storage/full/` 为占位骨架

### 2.2 一个设计裁决（detail.md 内部矛盾）

「Wiki 流程详解」是直写单表风格，「数据库表」章节给出 12 表三层 schema 且明示以其为准。

**裁决**：建全 12 表 SQLAlchemy 模型，管线按流程详解跑，发布层做"直通式"最小实现——每次 finalize 自动创建一条 release + release_pages + active_release 指针 + outbox 事件（无回滚 UI）。这样 schema 完整、行为等价于 WeKnora 的"写入即发布"。页面正文存 `wiki_page_revisions`（每次变更一条不可变版本）+ `wiki_chunks`（段落），`wiki_pages` 只存登记信息。

## 三、项目目录结构

```
pyrag/
├── AGENTS.md                      # 编码规范与约束（同级目录）
├── docs/ARCHITECTURE.md           # ← 本文件
├── pyproject.toml
├── config/
│   ├── settings.toml
│   └── prompts/                   # wiki.yaml graph.yaml generation.yaml vlm.yaml faq.yaml
├── src/pyrag/
│   ├── settings.py                # pydantic-settings，env 前缀 PYRAG_
│   ├── api/                       # FastAPI：app.py deps.py routers/{knowledge_bases,documents,retrieval,chat,wiki,graph}.py
│   ├── models/                    # SQLAlchemy：base.py knowledge_base.py knowledge.py chunk.py artifact.py wiki.py(12表)
│   ├── schemas/                   # Pydantic DTO
│   ├── storage/                   # ★ 抽象接口层
│   │   ├── interfaces.py          # VectorStore/KeywordIndex/GraphStore/CacheLock/ObjectStore Protocol
│   │   ├── factory.py             # profile=lite|full 组装
│   │   ├── vector_local.py        # numpy 暴力余弦（默认）
│   │   ├── vector_hnsw.py         # hnswlib 可选后端
│   │   ├── keyword_bm25s.py       # bm25s + jieba
│   │   ├── graph_local.py         # PMI 内存图 + JSON 快照
│   │   ├── lock_redis.py  lock_local.py  object_local.py
│   │   └── full/                  # vector_es.py keyword_es.py graph_neo4j.py（骨架 + NotImplementedError）
│   ├── llm/                       # client.py(Chat/Embedding/VLM) rerank.py retry.py json_repair.py
│   ├── parsing/                   # ← 移植 docreader
│   │   ├── base.py registry.py    # ParseResult(markdown+images)、魔数路由、链式解析
│   │   ├── pdf_parser.py docx_parser.py doc_parser.py
│   │   ├── excel_parser.py xlsx_repair.py xlsx_merge.py
│   │   ├── pptx_parser.py markdown_parser.py text_parser.py images.py
│   ├── chunking/                  # profiler.py heading.py heuristic.py recursive.py validator.py parent_child.py
│   ├── indexing/                  # embedder.py pipeline.py(幂等) image_pipeline.py(OCR/VLM级联)
│   ├── retrieval/                 # hybrid.py fusion.py(RRF) rerank_stage.py graph_boost.py faq.py
│   ├── generation/                # context.py(父子展开) citations.py answer.py(SSE)
│   ├── wiki/                      # queue.py debounce.py ingest.py(Map) merge.py(Reduce) finalize.py retract.py embed_pages.py build.py(同步编排)
│   ├── graph/                     # builder.py weights.py(PMI) chunk_graph.py mermaid.py snapshot.py
│   ├── tasks/                     # celery_app.py parse_tasks.py faq_tasks.py wiki_tasks.py graph_tasks.py
│   └── prompts/loader.py          # YAML + {{.Var}} 渲染
├── tests/                         # fixtures/ + 各模块单测 + conftest.py(内存SQLite、respx mock LLM、fakeredis)
└── scripts/                       # run_api.py run_worker.py init_db.py e2e.py
```

## 四、模块核心设计

### 4.1 解析层（parsing/）

- `ParseResult = {markdown: str, images: list[ImageRef(base64, uuid, mime)]}`，作为库被 Celery 任务直接调用
- `registry.py` 按魔数优先识别（OLE `D0CF11E0` 伪 docx→doc 链），复刻 docreader 的 chain 语义（首个成功即用）
- pdf 全部阈值按 detail.md 集中为模块常量：扫描页 0.5/10字符/0.1、标题行高比 2.0/1.6/1.35、页眉页脚 60%/4 页、嵌入图 80px/1%/50 张、质量回退 18%/12%/20%/8%
- 移植来源：`../WeKnora/docreader/parser/`

### 4.2 切块层（chunking/）

- 三层策略链 heading→heuristic→recursive + auto profiler
- validator 回退条件：0 块 / 大文档只产 1 块 / 小块（<50 字符）>1/4 且 >2 个 / 最大块 < chunkSize/4 / 任一块 > 2×chunkSize
- 6 组保护正则占位符法（代码块 / `$$` 公式 / 图片 / 链接 / 表格），保护块上限 7500 字符
- 父子分块：parent 4096 / child 384 / child overlap = 384//5
- 默认 chunk size 512 / overlap 80；分隔符优先级 `["\n\n", "\n", "。"]`
- 移植来源：`../WeKnora/internal/infrastructure/chunker/`

### 4.3 嵌入与索引（indexing/）

- **嵌入内容**：文档标题 + 标题面包屑路径（从根到当前块）+ 块内容
- **预处理**：超过 20000 字符截断
- **批处理**：嵌入批 5，索引批 40 并发 5
- **重试**：5 次指数退避，200ms 起
- **幂等序列**：软删旧 chunks → 删向量 → 删关键词 → 删图谱 → 写新 chunks → 嵌入子块 → 写索引 → 更新 `parse_status` → 投递 faq/wiki/graph 任务
- **图片级联**（image_pipeline.py）：OCR 优先 → VLM 兜底，按内容自然取舍（文字图走 OCR、语义图走描述）

### 4.4 检索（retrieval/）

- **并行通道**：向量 + BM25（asyncio.gather）
- **RRF 融合**：`score = vector_weight/(k+vector_rank) + keyword_weight/(k+keyword_rank)`，k=60，weights 0.7/0.3
- **分数归一化**：仅向量分钳位 [0,1]，BM25 不缩放（RRF 排名免疫）
- **重排序**：rerank TopK=10，threshold=0.2；阈值降级（无结果时降到 0.7×原阈值，下限 0.3）
- **Wiki 加权**：rerank 后对所有 `wiki_page` 切块 ×1.3
- **FAQ**：独立索引，迭代检索 ≤5 轮，起始 TopK=matchCount×3，按标签优先级排序
- **图谱增强**：查询实体抽取 → 图谱找相关 chunk → 并入候选去重（一跳 + 两跳，衰减 0.5）
- **返回结构**：`Candidate` 必须保留 chunk_id / score / 来源通道 / chunk_type / 标题路径 / document_id，不能过早降级为纯文本

### 4.5 生成（generation/）

- 父子块展开：text→parent_text（仅生成阶段，不影响向量检索）
- 邻居扩展（短上下文）
- XML 风格上下文组装 `<source type="..." priority="...">`
- 引用协议：`<ref id="cN"/>` / `<ref id="wN"/>` 内联标记
- SSE 流式：`message` / `references` / `done` 事件

### 4.6 Wiki 生成（wiki/）


| 常量                              | 值                              | 出处                      |
| --------------------------------- | ------------------------------- | ------------------------- |
| 防抖 ingest / followup / finalize | 30s / 5s / 20s                  | detail.md Wiki §步骤1    |
| 每批最多文档                      | 5                               | detail.md Wiki §步骤2    |
| Map / Reduce 并发                 | 10 / 10                         | detail.md Wiki §步骤3-4  |
| in-flight 槽位上限                | 4                               | detail.md Wiki §步骤2    |
| slug 锁 TTL / 等锁 / 轮询         | 5m / 2m / 50ms                  | detail.md Wiki §步骤4    |
| finalize 锁 TTL / 续约            | 60s / 20s                       | detail.md Wiki §步骤6    |
| 引用分类批 / 并发                 | 12000 runes / 4                 | detail.md Wiki §步骤3    |
| 抽取粒度                          | Focused / Standard / Exhaustive | detail.md Wiki §七类页面 |
| 死信阈值                          | 5 次                            | detail.md Wiki §步骤2    |
| LLM 重试                          | 3 次，2s 起指数退避             | detail.md Wiki §步骤2    |
| 最大原文                          | 32768 runes                     | detail.md Wiki §步骤3    |
| 去重 Jaccard / TopK / 旁路        | 0.08 / 5 / 25                   | detail.md Wiki §步骤4    |

- **Map**（单文档抽取）：拼原文(≤32768) → Pass0 候选 slug → 引用分类（按批 12000 字符，并发 4）→ 摘要 + 引用映射并行
- **Reduce**（跨文档归并）：per-slug SETNX 锁 → 去重（表面相似度预筛：字符 bigram Jaccard，topK=5、floor=0.08、语料≤25 页直通 → LLM WikiDeduplicationPrompt，合并项重定向到已有 slug 并追加别名）→ WikiPageModifyPrompt 合并 → 直接写 `wiki_pages`（无 revision/release，写入即发布）
- **Finalize**（全局终态）：首页引言（首建/增量）→ 分类分配 → Synthesis/Comparison 生成 → 交叉链接注入 → 死链清理
- **撤回**：DELETE /documents/{id} 软删文档与切块、清理向量/关键词索引，入队 retract op（同队列 op='retract'）→ SourceRefs 定位受影响页面 → LLM 重写剩余内容 → 唯一来源页软删归档 → 触发 finalize
- **重解析 replace**：解析完成显式入队本文档；reduce 前比对旧 source_refs 页面集，不再出现的 slug 撤回，重叠 slug 的合并提示词携带 `<replaced_document>` 指令替换旧贡献
- **概览页 slug**：`overview/<doc_id>`（title 为文档标题）。与 detail.md 字面 `overview/<文档标题>` 不同，避免同名文档撞页
- **Synthesis/Comparison**（设计裁决：detail.md §C 与§流程详解注矛盾，pyrag 无 Agent 工具体系，采 finalize 自动生成）：Synthesis 需 `kb.wiki_config.synthesis_model`（未配置跳过），输入全库已发布页面正文（上限 100 页）；Comparison 用默认模型，选 out_links 共现≥2 的实体/概念对，上限 3 组
- **Index 导航**：index 页 content 只存 LLM 引言；目录列表由 `GET /wiki/{kb_id}/index` 程序化组装（按类型分组 + folders 树），不靠 LLM
- **FAQ**：`kb.question_generation_config.enabled` 开启后嵌入完成异步为每个子切块生成 3-5 问句（chunk_type=faq，并发 4）；手动导入 API `POST /faq/{kb_id}/items`（标准问+相似问分别嵌入，反例问仅存储不做抑制）；FAQ 类知识库检索走独立路径按标签优先级排序，不参与融合与 wiki 加权

**精简后的表**：`wiki_pending_operations` / `wiki_dead_letters` / `wiki_pages` / `wiki_folders`。已删除未使用的发布层 10 表（wiki_build_requests / wiki_locks / wiki_chunks / wiki_source_refs / wiki_page_revisions / wiki_releases / wiki_release_pages / wiki_active_releases / wiki_page_links / wiki_publish_outbox）——写入即发布，无 release 机制（detail.md §步骤5）。

**七类页面**：overview（文档级） / concept / entity（概念与实体） / index（知识库级） / synthesis（综览） / comparison（对照） / faq（独立路径）

**提示词全套**：`config/prompts/wiki.yaml` 移植自 `../WeKnora/internal/agent/prompts_wiki.go`，经 `prompts/loader.get_prompt` 注入，代码内不硬编码，包含：

- WikiSummaryPrompt / WikiCandidateSlugPrompt / WikiChunkCitationPrompt
- WikiPageModifyPrompt / WikiDeduplicationPrompt
- WikiIndexIntroPrompt / WikiIndexIntroUpdatePrompt / WikiTaxonomyPlanPrompt
- WikiSynthesisPrompt / WikiComparisonPrompt
- WikiGranularityGuidance 三档

### 4.7 知识图谱（graph/）

**分层实现**：

- **轻量版（本地开发）**：PMI 内存图版（按 detail.md 实现）
- **完整版（生产环境）**：Neo4j 图数据库（与 WeKnora 当前实际接入一致）

通过 `storage/interfaces.py` 的 `GraphStore` Protocol 抽象，轻量版与完整版实现相同接口，业务代码无需修改。

#### 4.7.1 轻量版：PMI 内存图

- **实体抽取**：逐 chunk LLM 调用，并发 4；输出 `{title, type, description}` JSON 数组
- **关系抽取**：每 5 chunk 一组，并发 4；输入该组所有实体+描述，输出 `{source, target, description, strength}` JSON 数组；实体<2 跳过该组
- **去重**：内存 `entityMap`（同名追加 ChunkIDs，频次+1）、`relationMap`（源#目标为键，描述/强度覆盖、源 chunks 合并）
- **PMI 权重计算**：
  - `P(source,target) = relFreq / totalRelOccurrences`
  - `P(x) = entityFrequency(x) / totalEntityOccurrences`
  - `PMI = max(log2(P(x,y) / (P(x)*P(y))), 0)`
  - `combinedWeight = 0.6 × normalizedPMI + 0.4 × normalizedStrength`
  - `weight = 1.0 + 9.0 × combinedWeight` → 映射 [1, 10]
- **度数**：入度 + 出度；关系 `CombinedDegree` = 两端实体度数之和
- **切块关联图**：基于共同出现的实体对建立 chunk↔chunk 关联
- **查询**：一跳（按 weight→degree 排序 topK）；两跳（两段权重乘积 × 衰减 0.5）
- **Mermaid**：DFS 连通分量 → subgraph；Strength>7 用 `==>`；Frequency>5 高亮
- **存储**：内存 + JSON 快照制品（signature = chunks content_hash 拼接）
- **失败容忍**：单组关系抽取失败跳过该组

#### 4.7.2 完整版：Neo4j

- **实体节点**：label = `ENTITY<kb_id>:ENTITY<knowledge_id>`，属性含 `chunks` 数组（关联 chunk_id 列表）
- **关系边**：label = `RELATES_TO`，属性含 `weight`（PMI 计算后写入）、`description`、`chunks`
- **写入**：`apoc.merge.node` / `apoc.merge.relationship`（幂等合并）
- **查询**：Cypher 直接写一跳/两跳路径匹配，无需内存 PMI 计算（权重从关系属性读取）
- **索引**：节点按 `title` 建索引，加速实体查找

#### 4.7.3 GraphStore 接口

```python
class GraphStore(Protocol):
    async def add_graph(self, kb_id: str, knowledge_id: str, graph: ExtractedGraph) -> None
    async def delete_document(self, kb_id: str, knowledge_id: str) -> None
    async def search_entities(self, kb_id: str, entity_names: list[str], top_k: int) -> list[ChunkHit]
    async def get_relations(self, kb_id: str, entity_names: list[str], hops: int = 1) -> list[RelationHit]
    async def snapshot(self, kb_id: str) -> bytes | None  # 仅轻量版返回 JSON 快照
    async def load_snapshot(self, kb_id: str, data: bytes) -> None  # 仅轻量版
```

轻量版实现 `storage/graph_local.py`（内存 + JSON 快照），完整版实现 `storage/full/graph_neo4j.py`（neo4j Python driver）。

#### 4.7.4 切换路径

1. `storage/factory.py` 按 `graph.profile = "lite" | "neo4j"` 组装
2. 完整版需配置 `graph.neo4j.uri` / `graph.neo4j.user` / `graph.neo4j.password`
3. 轻量版启动时从 JSON 快照加载（若存在），否则空图；完整版直接连 Neo4j
4. 切换后需重建图谱（`scripts/rebuild_graph.py`）

### 4.8 数据库表（详见 detail.md §数据库表）

- **关系型**：knowledge_base / knowledge / artifact / chunks / 12 张 wiki_* 表
- **ES 索引**（完整版）：dense_vector(dims) + content (BM25)
- **图谱**：内存 + JSON 快照
- **Redis**：嵌入缓存 / 重排缓存 / 会话 / 分布式锁

## 五、实施

里程碑


| 阶段                  | 内容                                                                | 验收标准                                                                                                     |
| --------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **M1 解析**           | 六类 parser 移植 + knowledge/artifact 模型 + 上传接口（同步版）     | fixtures 各格式出正确 markdown+图片；魔数识别用例通过；LibreOffice 缺失时降级报错而非崩溃                    |
| **M2 切块+嵌入+存储** | chunking + indexing + storage + Celery 异步化 + OCR/VLM 级联        | 三层策略选择与回退单测；幂等重解析；嵌入内容含面包屑前缀（respx 抓包断言）                                   |
| **M3 检索+生成**      | RRF、rerank、FAQ 迭代、父子展开、SSE 引用回答                       | BM25/向量各自命中用例；RRF 排序手算一致；`<ref id="cN"/>` 可回链；空召回返回"无法回答"                       |
| **M4 Wiki**           | 12 表 + 防抖队列 + Map/Reduce + finalize + 撤回 + wiki_page 入索引  | 3 篇交集文档产出归并页（同名实体单页多 SourceRefs）；删文档触发重写/软删；wiki_page ×1.3 生效；死信路径单测 |
| **M5 图谱**           | 构建 + PMI + 关联图 + 快照 + Mermaid + 检索接入                     | mock LLM 下权重/度数手算一致（黄金用例）；两跳衰减排序正确；图谱通道召回字面不匹配用例                       |
| **M6 集成收尾**       | retrieval profile 参数、Redis 缓存、full/ 占位、AGENTS.md、e2e 脚本 | `scripts/e2e.py` 一键全链路；Windows solo pool 可跑；AGENTS.md 可定位切换点                                  |

## 六、关键配置默认值


| 配置                         | 默认值                          | 出处                 |
| ---------------------------- | ------------------------------- | -------------------- |
| ChunkSize / Overlap          | 512 / 80                        | detail.md §切块     |
| Parent / Child chunk         | 4096 / 384                      | detail.md §父子分块 |
| 保护块/块绝对上限            | 7500 字符                       | detail.md §第三层   |
| RRF k                        | 60                              | detail.md §混合检索 |
| RRF 权重                     | vector 0.7 / keyword 0.3        | detail.md §混合检索 |
| EmbeddingTopK                | 50                              | detail.md §混合检索 |
| Vector / Keyword 阈值        | 0.15 / 0.3                      | detail.md §混合检索 |
| RerankTopK / RerankThreshold | 10 / 0.2                        | detail.md §混合检索 |
| wiki boost                   | ×1.3                           | detail.md §混合检索 |
| 嵌入重试                     | 5 次，200ms 起指数退避          | detail.md §嵌入     |
| BATCH_EMBED_SIZE             | 5                               | detail.md §嵌入     |
| FAQ 迭代检索                 | ≤5 轮，起始 TopK=matchCount×3 | detail.md §FAQ      |
| 实体/关系抽取并发            | 4 / 4                           | detail.md §知识图谱 |
| 关系批大小                   | 5 chunk                         | detail.md §知识图谱 |
| PMI/Strength 权重            | 0.6 / 0.4                       | detail.md §知识图谱 |
| 两跳衰减                     | 0.5                             | detail.md §知识图谱 |
| Wiki 防抖                    | 30s / 5s / 20s                  | detail.md §Wiki     |
| Wiki 死信                    | 5 次                            | detail.md §Wiki     |
| Wiki LLM 重试                | 3 次，2s 起                     | detail.md §Wiki     |

## 七、依赖清单

**核心**：fastapi, uvicorn, sse-starlette, pydantic, pydantic-settings, sqlalchemy>=2.0, alembic, httpx, celery[redis], redis, tenacity

**解析**：pypdfium2, markitdown, python-docx, openpyxl, pandas, Pillow, python-pptx, filetype

**检索**：numpy, bm25s, jieba；可选 hnswlib

**工具**：json-repair, PyYAML, structlog

**测试**：pytest, pytest-asyncio, respx, fakeredis

**可选系统依赖**：LibreOffice(soffice) / antiword / ImageMagick

## 八、风险与取舍

1. **Windows + Celery**：prefork 池在 Windows 不可用。M2 起 worker 统一 `--pool=solo`（开发）或 `--pool=threads`（IO 密集的 LLM 调用够用）；生产建议 Linux/WSL。任务函数不依赖 fork 语义。
2. **docx 多进程分页**：Windows spawn 每个子进程重导入很重。默认串行，`parsing.docx.multiprocess=false`；仅 >1000 段的大文档且非 Windows 时启用，且入口守护齐全。
3. **LibreOffice / antiword / ImageMagick 可选化**：启动时能力探测缓存结果；缺失时对应格式返回带指引的 `ParserCapabilityError`，写入 `knowledge.error_message`，不影响其他格式。
4. **bm25s 增量**：无真增删，取"标脏 + KB 级重建"。chunks 表为唯一真值；超大 KB 重建变慢时切完整版 ES 即可消除（详见 `AGENTS.md` 切换说明）。
5. **向量维度与内存**：Qwen3-Embedding-4B 输出 2560 维，10 万子块×2560 维约 0.5GB(float16)，单查询暴力点积 <100ms。若实际维度不同，以首次嵌入返回为准写入 KB 配置并校验一致性（跨库检索要求同模型）。
6. **LLM JSON 稳定性**（Wiki/Graph 大量结构化输出）：统一 json_repair + schema 校验 + 重试；关系抽取失败按 detail.md 跳过该组不中断。
7. **锁语义降级**：Redis 宕机时 per-slug 锁按 detail.md 宽容失败（不锁直写，终态可修复）；本地开发可用 `lock_local.py`（进程内）替代——但多 worker 场景必须 Redis。
8. **发布层简化**：直通 release 牺牲了回滚能力，换取与 detail.md 流程节行为一致；完整 release 切换留作 full 档演进项。

## 九、变更记录


| 日期       | 变更                                                                                     | 影响                        |
| ---------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| 2026-07-25 | 初版（按 detail.md + WeKnora 参考实现设计）                                              | M1–M6 全部未开始            |
| 2026-07-25 | §4.7 图谱分层：轻量版 PMI 内存图（本地开发）+ 完整版 Neo4j（生产）；新增 GraphStore 接口 | M5 图谱阶段实现需按分层设计 |
| 2026-07-26 | Wiki 管线补全：去重接线（预筛 topK=5/floor=0.08/≤25 页直通）、retract 全链路、重解析 replace、Index 导航 API、Synthesis/Comparison 终态生成（共现≥2、上限 3 组）、FAQ 全路径、提示词 YAML 化、删除未用发布层 10 表；overview slug 改 doc_id | wiki/*、api/routers/{documents,wiki,faq,retrieval}.py、models/wiki.py、migrations |
| 2026-07-26 | 前端三大功能补全：新增 chunks/wiki/graph API 端点、wiki_build_task 接线、图谱 API 读快照；前端 3 视图 3 路由 | B1-B5 F1-F4 |
| 2026-07-26 | Wiki 编排修复：新增 `wiki/build.py::run_wiki_build` 统一三处编排（Celery 任务/手动 API/同步上传路径）；修复防抖时序（构建即触发者，入队后 available_at 前移）、doc_title 误传 UUID、未处理检测改为 completed_at 过滤、wiki API 响应模型 updated_at 改 datetime | wiki/、tasks/、api/routers/{wiki,documents}.py |
| 2026-07-26 | 数据一致性修复：文件型 SQLite 弃用 StaticPool 改默认连接池（单连接跨线程共享会被 API 请求的 ROLLBACK 吞掉后台线程未提交写入，导致切块/图谱/Wiki 全空），新增 busy_timeout=30s；RedisLock 连接失败降级 LocalLock；新增 GET /documents/{id}/content 按切块偏移重建解析后文本；前端侧边栏新增 Wiki/知识图谱入口（顶级路由 + 知识库选择器），切片页改左右分栏源文高亮对照 | models/database.py、storage/lock_redis.py、api/routers/documents.py、前端 router/layouts/views |
| 2026-07-26 | 父子切块语义修正：父块 = 标题层级完整 section（split_by_heading 新增 whole_sections 模式，取消 4096 截断与贪心细分），无标题文档退化为 parent_size 粒度递归切分；_greedy_split 代码围栏原子化（mermaid 等不再被空行拆碎） | chunking/{heading,parent_child,__init__}.py |
| 2026-07-26 | 孤立标题块修复：split_by_heading 新增 _merge_heading_only 后置合并（参考 WeKnora chunker coalesceTinyChunks：仅原文连续时合并，以原文切片保持 span 不变量），仅含标题行的块并入下一块（文末则向前并），覆盖 _greedy_split 三条泄漏路径（围栏前 flush、大段落硬切前 flush、溢出 flush）及空 section 父块 | chunking/heading.py |
| 2026-07-26 | 删除文档仍被检索召回修复：DELETE /documents/{id} 联动软删 chunks + 清理向量/关键词/图谱索引（GraphManager 新增 delete_document）；DELETE /kb/{id} 联动软删 chunks；expand_parent_chunks 增加 Knowledge.deleted_at 兜底过滤历史残留；LocalVectorStore meta 落盘（meta.json）、BM25KeywordIndex corpus/meta 落盘（corpus.json + indexed_ids.json 一致性校验），修复重启后按文档删除失效与 BM25 检索全空 | api/routers/{documents,knowledge_bases}.py、storage/{vector_local,keyword_bm25s}.py、graph/builder.py、generation/context.py |

---

**维护规则**：

- 任何架构变更（新增模块、修改接口、调整阈值、切换依赖）必须更新本文件对应章节
- 同步在 commit message 中引用本文件章节号
- 参数默认值修改必须同步更新第六章表格并注明出处
