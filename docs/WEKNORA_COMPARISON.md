# WeKnora vs KnowLink 对比报告

> 生成日期：2026-07-26
> 对比范围：**RAG（文档入库→检索→问答）+ Wiki + 知识图谱**
> 对比原则：KnowLink 为刻意的本地轻量版（SQLite + numpy + bm25s + PMI 内存图），**不把「用了 Elasticsearch / Neo4j / MySQL 等重型组件」当差距**，只看两类问题：
> 1. WeKnora 有、KnowLink 没有的**功能能力**（lite 架构同样能实现的）
> 2. 双方都实现的功能上，WeKnora **实现质量更高**（鲁棒性 / 性能 / 效果）的地方
>
> 调研方法：三路并行代码级调研（功能覆盖 / 入库链路 / 检索问答图谱链路），关键结论已逐条对源码复核。所有断言均附证据文件路径。

---

## 一、功能差距：WeKnora 有、KnowLink 没有

按对 RAG 效果的价值排序。「lite 可落地」表示在当前 SQLite/numpy/bm25s 架构下无需新增重型组件即可实现。

| # | 差距 | 说明 | WeKnora 证据 | KnowLink 验证依据 | lite 可落地 |
|---|---|---|---|---|---|
| 1 | **多轮查询改写 + 意图分类** | LLM 低温（0.3）改写 + 意图分类，限 150 token；模型三级回退（vision→VLM→文本）；JSON 解析失败时把原文当改写结果兜底。KnowLink 检索阶段完全看不到 history（全项目 grep 无 rewrite 代码），多轮指代（"它的价格呢"）必然召回失败 | `WeKnora/internal/application/service/chat_pipeline/query_understand.go` | `src/knowlink/api/routers/chat.py` L33-47 直接用原 query 检索 | ✅ 一次 LLM 调用，有 history 才触发 |
| 2 | **服务端会话/消息持久化** | session/message 落库、会话标题自动生成、历史按 MaxRounds 截断加载、检索上下文回写用户消息供后续轮次使用。KnowLink 无 session/message 表，history 由前端透传且无长度控制 | `chat_pipeline/load_history.go`、`into_chat_message.go` L204-233、`handler/session/` | `src/knowlink/models/` 无 Session/Message 模型 | ✅ SQLite 两张表 + 最近 N 轮窗口 |
| 3 | **检索评估体系** | QA 数据集驱动的端到端检索/生成评估（执行评估任务 + 结果查询）。KnowLink 检索参数可全套覆写（`retrieval.py` 10 项参数），却无任何量化评估手段，调参无依据 | `internal/handler/evaluation.go`、顶层 `dataset/` | 全项目 grep evaluat/dataset 零命中 | ✅ 高价值 |
| 4 | **Chunk 编辑/删除** | 单块更新、删除、按知识删除全部块。KnowLink chunks 只读列表，坏分块只能整文档重解析 | `internal/router/router.go` L307-322 | `src/knowlink/api/routers/documents.py` 仅 `GET /{doc_id}/chunks` | ✅ 直接影响检索质量修复效率 |
| 5 | **无 LLM 本地 query 扩展** | 初检召回不足时生成至多 5 个查询变体（去停用词/去疑问词/引号短语/分隔切段），并发补检索且关键词阈值放宽 0.8 倍。纯字符串处理，零 LLM 成本 | `chat_pipeline/query_expansion.go` L15-170 | KnowLink 无对应物 | ✅ 零成本 |
| 6 | **URL/网页入库 + 更多格式** | `POST kb/url` 网页抓取入库（web/mhtml parser）；格式覆盖 epub/mhtml/csv/doc(OLE)/独立图片文件（jpg/png/gif/bmp/tiff/webp 走 ImageParser）；markitdown 兜底引擎 + opendataloader 可选引擎（带可用性探测）。KnowLink 仅 pdf/docx/xlsx/pptx/text 五类，无独立图片文件解析器 | `handler/knowledge.go` L438、`docreader/parser/registry.py` L118-166 | `src/knowlink/parsing/__init__.py`、`registry.py` | ✅ |
| 7 | **建议问题/追问推荐** | 回答后生成推荐追问 | `handler/session/message_suggestion.go` | grep suggest 零命中 | ✅ 低成本高感知 |
| 8 | **FAQ 高级管理** | 批量 upsert（含 dry_run 校验）、批量改标签/字段、批量删除、FAQ 混合搜索、CSV/JSON 导出、导入进度、追加相似问。KnowLink 仅单条导入/列表/删除 | `handler/faq.go`（13 个端点） | `src/knowlink/api/routers/faq.py`（3 个端点） | ✅ |
| 9 | **Wiki 手动编辑与目录组织** | 页面创建/更新、文件夹 CRUD、页面移动、子树路径级联重算。KnowLink wiki 只读（index/pages）+ build/status，页面不可编辑、无文件夹 | `handler/wiki_page.go` | `src/knowlink/api/routers/wiki.py` | ✅ |
| 10 | **分块调试预览 / 解析 Span 树** | 入库前预览分块策略效果（chunker_debug）；解析各阶段 span 树（状态/耗时/错误码，支持查看历史尝试）。KnowLink 仅 parse_status 单字段 + SSE | `handler/knowledge.go` L642（GetKnowledgeSpans）、`chunker_debug.go` | KnowLink 仅 `/status` | ✅ |
| 11 | 手工创建/在线编辑知识条目 | Markdown 手工录入 + 更新 | `handler/knowledge.go` L536/L1577 | KnowLink 仅 upload/import-local | ✅ |
| 12 | 知识批量操作 | 批量删除、清空知识库、知识属性更新、浏览器内嵌预览 | `handler/knowledge.go` L1081/L1185/L1332 | KnowLink 单个删除/重命名/下载 | ✅ |
| 13 | 聊天内临时附件/图片问答 | 会话级临时文档、图片上传分析 | `handler/session/`（temporary_document、image_upload） | KnowLink 聊天仅纯文本 | ✅ |
| 14 | 知识级标签体系 | 独立标签 CRUD 挂载到 knowledge，检索可按 tag 圈定范围 | `handler/tag.go` | KnowLink tags 仅存在于 FAQ metadata | ✅ |
| 15 | 数据源连接器 + 定时同步 | 飞书 wiki 等外部源的资源树浏览、定时/手动同步、同步日志 | `handler/datasource.go` | KnowLink 无 | ⚠️ 连接器生态重；「本地目录/URL 定时同步」简化版可做 |

**范围外功能**（一句话清单，不深入）：Agent 自定义编排、MCP 服务、技能系统、IM 集成、网页嵌入问答挂件、多租户/RBAC/审计、Web 搜索、模型运行时管理、小程序、桌面端。

### 最关键的 5 个差距（按价值排序）

1. **多轮查询改写 + 会话持久化**（#1+#2）——直接决定多轮对话检索命中率，lite 成本极低
2. **评估体系**（#3）——没有评估就无法量化任何调参/改进的收益
3. **Chunk 编辑**（#4）——坏分块的最短修复路径
4. **URL 入库 + 格式覆盖**（#6）——扩大知识来源
5. **建议问题 + 本地 query 扩展**（#5+#7）——低成本的体验与召回提升

---

## 二、同功能实现质量对比：WeKnora 做得更好的地方

### 入库链路（解析→分块→嵌入→索引→任务编排）

#### 1. 表头跨块追踪与重注入（对表格问答价值最高）

- **WeKnora**：`docreader/splitter/header_hook.py` 的 HeaderTracker 状态机——跨块追踪活跃 markdown 表头，新块头部条件性注入「表头行+分隔行」；空表头行用首个数据行补全列名；列数不匹配自动终止追踪；Go 侧 `internal/infrastructure/chunker/header_tracker.go` 同构实现。
- **KnowLink**：分块仅有代码围栏原子性（`chunking/heading.py` `_split_respecting_fences`）与公式块回并，**表格零保护**——长表格被按段落切开后，后续块完全丢失列名语义，表格类问题的召回与生成质量直接受损。
- **借鉴**：移植 HeaderTracker（表格跨块时把表头前缀注入后续块），至少先把 markdown 表整体作为 protected 段。

#### 2. 索引写入的失败清理一致性

- **WeKnora**：`internal/application/service/knowledge_process.go` L587-636——BatchIndex 失败时同时删除已写 chunks 和已写索引；索引前后两次检查 deleting/cancelled 状态并差异化清理（deleting 删数据，cancelled 保留已持久化部分）。
- **KnowLink**：`src/knowlink/indexing/pipeline.py` L42-50 **先删旧索引再分块嵌入重建**——中途失败（嵌入超时/LLM 熔断）会让文档处于「旧索引已删、新索引半写或没有」的不可检索状态，且无法回滚。
- **借鉴**：改为先写新索引（版本号/临时命名空间）成功后再删旧；或失败时清理半写的新向量。

#### 3. 任务编排体系

- **WeKnora**：`internal/types/task.go` L96-178 多队列硬隔离（core/postprocess/enrichment/maintenance/wiki 各自独立并发数，env 可调）；`task_dead_letter.go` 死信落库可人工重放；可重试/永久失败显式分类（unmarshal 错误、内容不足等 return nil 不重试）；阶段状态机支持断点续传（重试跳过已完成阶段）；ParseStatus 含 deleting/cancelled，删除与解析中任务无竞态。
- **KnowLink**：单 Celery 队列；`tasks/parse_tasks.py` max_retries=2 固定 5s 间隔且**整个任务（含重新解析）从头重跑**；无死信（重试耗尽仅 status=failed + traceback 前 2000 字）；无取消状态——删除文档与正在跑的解析任务存在竞态；graph/wiki 子任务异常直接吞掉不重试。Celery 任务无 time_limit（仅 soffice 有 120s 超时）。
- **借鉴**：Celery 分 ingest/enrich 两个队列即可拿到 80% 收益；重试改指数退避并缓存已解析的 markdown（重试只重跑索引）；加 failed_tasks 表支持一键重试；管线关键节点查一次 `Knowledge.deleted_at` 命中即中止。

#### 4. 解析层鲁棒性体系

- **WeKnora**：`docreader/parser/chain_parser.py` FirstParser 责任链——单解析器异常自动换下一个，直到产出有效文档；`xlsx_repair.py` 主动修复损坏 xlsx；`pdf_parser.py` L34-43 pdfium 全局锁（pdfium 非线程安全，并发解析会死锁整个进程）+ `concurrency.py` 重型解析器进程级信号量；全部阈值 env 可配（L76-97）。
- **KnowLink**：每格式单解析器路径（`parsing/registry.py`），解析器崩溃即整个文件失败；阈值全部硬编码（`pdf_parser.py` L13-39）；无解析并发限制；**`_detect_columns_and_reorder` 是空实现（L242-243），双栏 PDF 实际未做分栏处理**；解析出空内容静默返回空 markdown 而非报错。
- **借鉴**：docx/pdf 加「主解析器异常→备选解析器」责任链（markitdown 作兜底）；阈值配置化；补真实分栏逻辑或在文档说明该限制。

#### 5. VLM 图片处理可靠性

- **WeKnora**：`internal/application/service/image_multimodal.go`——每图独立异步任务；孤儿任务检测（知识已删则丢弃不重试，L369-401）；finalize 门闩保证单图永久失败不会把整个文档卡在 processing（L196-222，历史上「卡解析」的第一大原因）；末次尝试识别（L403-414）。
- **KnowLink**：`src/knowlink/indexing/image_pipeline.py`——VLM/OCR 失败一律裸 `except: pass` 静默吞掉（L38-39、L46-47、L68-69），**无重试、无失败记录（哪张图失败、为何失败不可知）、无按图片 hash 的结果缓存**（重解析重复扣费）。
- **借鉴**：VLM 调用加 tenacity 重试 + 失败落日志/metadata；按图片 content-hash 缓存 VLM 结果。

#### 其余入库链路对比（基本打平或各有所长）

| 维度 | 结论 |
|---|---|
| 图片过滤阈值（80px/1% 面积/50 张上限/0.5 跨页重复） | 两边数值一致（KnowLink 有意对标），WeKnora 可配置、KnowLink 硬编码 |
| 分块默认值（512/80 overlap） | 一致；但 KnowLink heading 策略路径无 overlap，WeKnora 全路径一致且可还原验证 |
| 标题上下文注入 | 等价设计（ContextHeader vs title_path 元数据 + 嵌入时拼接）；WeKnora 大节细分时子块取「起点处最深活跃标题」，KnowLink 子块共用 section 级 title_path，粒度略粗 |
| 文件去重幂等 | 打平（KnowLink sha256 + chunk 级 content_hash 略细；但 content_hash 只存不用，可用于重解析跳过未变块） |
| 进度上报 | WeKnora 阶段状态机粒度细；KnowLink 固定 10/30/50/100 四个点，嵌入 50% 后无中间进度 |

### 检索→重排→生成链路

#### 6. 深度去重

- **WeKnora**：`chat_pipeline/search.go` L169-289——chunk_id 去重 + 归一化内容签名去重 + `removePartialOverlaps`（短文本被高分块包含、或 token 重叠系数 ≥0.85 即剔除低分者）；`merge.go` 八步合并管线（去重→历史注入→父块解析→重叠区间合并→FAQ 填充→邻块扩展→再合并→终去重）。
- **KnowLink**：仅 chunk_id 级去重（`retrieval/fusion.py`）。**父块扩展后，同一父块的多个子块命中会解析成相同父内容，重复占用上下文窗口与 top_k 名额**。
- **借鉴**：`expand_parent_chunks` 之后加一层归一化内容签名 + 包含关系去重（几十行纯 Python）。

#### 7. 重排效果层

- **WeKnora**：`chat_pipeline/rerank.go`——最终分 = 复合分（rerank 模型分与检索 base 分融合，L199-203）+ FAQ boost（上限 1.0）+ MMR 多样性选择（lambda 0.7，L223）；空 passage 先剔除。
- **KnowLink**：`retrieval/rerank_stage.py`——rerank 分直接**覆盖**检索分，检索通道信号全部丢失；无 MMR（top_k 内容易同质化）；`llm/rerank.py` 单次 HTTP 无重试、无批处理分片（候选多时可能超 API 长度限制）。
- **借鉴**：最终分混入归一化检索分（如 0.8×model + 0.2×base）；候选按 32/64 分批调用；加简单 MMR（可复用 #6 的内容签名）。

#### 8. 流式容错

- **WeKnora**：`chat_pipeline/chat_completion_stream.go` L113-236——模型错误以独立 error 事件下发，不污染已输出答案；ctx 取消与 channel 关闭都走 flushDecoders，防止跨 chunk 边界的引用标记丢失；thinking/answer 双通道。
- **KnowLink**：`src/knowlink/generation/answer.py` L44-48——流中途异常直接 `yield "I cannot answer..."`，**若已输出半截答案，用户会看到答案后接一句自相矛盾的兜底话术**，且前端无法区分「正常内容」与「错误」。
- **借鉴**：异常改发独立 `error` 事件类型并保留已输出内容，前端按事件类型展示。

#### 9. 图谱 / FAQ 未接入聊天链路（复核实锤，KnowLink 侧问题最大的一条）

- **WeKnora**：图谱是问答管线的一等插件——`chat_pipeline/search_entity.go` 在 ENTITY_SEARCH 事件对启用图谱的 KB 并发查图，命中的新 chunk（filterSeenChunk 去重）补入候选集，**与其他候选一起过 rerank/merge，图谱召回真正参与最终答案**。FAQ 同理（FAQ 分离 + 高置信直答标注进 prompt，`into_chat_message.go` L41-67）。
- **KnowLink**（grep 复核确认）：`faq_search` 与 `graph_boost` **只接在 `/retrieval` 检索测试路由（`api/routers/retrieval.py` L87/L113），聊天路由 `chat.py` 两者都没接**——用户在检索测试页看到的 FAQ 分流和图谱召回，在真正聊天时并不生效。另外 `retrieval/graph_boost.py` L38-41 存在死代码：把**字符串 query** 当向量传给 `storage.vector.search(kb_id, query, top_k=0)` 且结果从未使用，一旦该调用抛异常整个 graph_boost 返回空——图谱召回可能被这行无意义调用直接短路。
- **借鉴**：chat 路由在 rerank 前并入 faq_search 与 graph_boost 候选（复用 existing_ids 去重）；删除 graph_boost 中的无效 vector.search 调用。

#### 10. 多 KB 检索并发

- **WeKnora**：`search.go` L317-461 按嵌入模型身份分组，同组只算一次 query 向量、全库目标合并单次检索，KB 与 Web 检索 goroutine 并发。
- **KnowLink**：`chat.py` L40-43 多 KB **串行 await 循环**（KnowLink 单嵌入模型，无需分组，但并发是免费的）。
- **借鉴**：改 `asyncio.gather` 一行收益。

---

## 三、KnowLink 反而更优的点（公平对比）

1. **PDF 文字层重建深度**（`parsing/pdf_parser.py`）：逐字符行重建（空格推断/标题按字号+中文模式分级/代码行号剔除/图表标签过滤）+ 五重质量指标自动回退原始文本（L142-197）+ 代码块自动围栏 + 公式页/扫描页整页转 VLM + 矢量图区域检测。WeKnora docreader 的 PDF 解析自述「刻意保持简单」，这些能力它没有。
2. **VLM OCR 级联内联回填**（`indexing/image_pipeline.py`）：OCR-first，有文字直接抽成 Markdown 替换占位符，图片内容成为一等检索文本；WeKnora 的 caption 只是补充描述，信息量通常低于 OCR 全文。
3. **父子块检索模型**（`chunking/parent_child.py`）：子块嵌入、命中返回父块，长文档上下文完整性优于 WeKnora 的 ContextHeader 面包屑方案（WeKnora 无父子块模型）。
4. **分块结果驱动的策略回退链**（`chunking/__init__.py` L38-48）：validate 不过按 heading→heuristic→recursive 换策略重切；WeKnora 的 strategy resolver 只做事前选择，无结果驱动降级。
5. **标准加权 RRF**（`retrieval/fusion.py`）：教科书式 rank-based 融合且通道权重可配，无需分数归一化假设；WeKnora 是分数域融合。
6. **图谱 reconcile 对账**（`graph/builder.py` L262-297）：以 DB 活块集为真值清理死实体/关系并重算 PMI；WeKnora 无对等物。
7. **rerank 阈值兜底更省钱**（`retrieval/rerank_stage.py` L40-42）：本地用降级阈值重过滤已有分数；WeKnora 用相同输入**二次调用** rerank 模型（`rerank.go` L148-179），多花一次 API 且结果确定性相同。
8. **嵌入批粒度重试**（`llm/retry.py` + `indexing/embedder.py`）：每批独立 tenacity 指数退避；WeKnora 把 BatchIndex 当整体失败处理。
9. **无 Redis 同步降级**（`tasks/parse_tasks.py` L19-48）：Redis 不可用时同步解析 + 进度内存回退，符合 lite 部署定位；WeKnora 强依赖 Redis/asynq。
10. **FAQ 迭代扩容检索**（`retrieval/faq.py`）：top_k 逐轮放大直到凑够 match_count + 标签优先级排序；生成上下文的 source XML 带 priority 分数与 title_path，给 LLM 的可信度线索多于 WeKnora 的简单 context id。

---

## 四、改进优先级建议

> 仅为排期参考，本报告不附带代码变更。

### P0 —— 直接影响问答正确性（聊天链路当前有断点）

| 项 | 内容 | 工作量 |
|---|---|---|
| chat 接入 FAQ 分流 + graph_boost | 把 `retrieval.py` 已有的 FAQ/图谱逻辑移入 chat 链路，删除 `graph_boost.py` L38-41 死代码 | 小 |
| 多轮查询改写 | 有 history 才触发一次低温 LLM 改写，失败回退原 query（参考 `query_understand.go` 的兜底解析） | 小 |
| 流式错误事件化 | `answer.py` 异常改发独立 error 事件，保留已输出内容 | 小 |

### P1 —— 影响检索质量

| 项 | 内容 | 工作量 |
|---|---|---|
| 表头跨块追踪注入 | 移植 HeaderTracker 到 `chunking/`，表格跨块时注入表头前缀 | 中 |
| 内容签名去重 | 父块扩展后加归一化签名 + 包含关系去重 | 小 |
| 索引切换先写后删 | `pipeline.py` 消除「文档无索引」窗口 | 中 |
| rerank 复合分 + 分批 | 混入检索 base 分；候选分批调用防超长 | 小 |
| 多 KB 并发检索 | `chat.py` 串行循环改 `asyncio.gather` | 极小 |

### P2 —— 工程健壮性

会话/消息持久化（SQLite 两张表 + 最近 N 轮窗口）、黄金评估集（复用 `tests/golden/` 机制扩展为端到端检索评估）、Chunk 编辑/删除 API + 前端、VLM 失败记录 + 重试 + 按 hash 缓存、Celery 双队列 + soft_time_limit、解析责任链兜底（markitdown）。

### P3 —— 覆盖面

URL/网页入库、独立图片文件解析、epub/mhtml/csv 格式、Wiki 页面编辑与文件夹、建议问题、本地 query 扩展、FAQ 批量管理/导出、分块调试预览。

---

## 附：调研证据文件索引

**WeKnora 侧**：
- 路由/功能全集：`internal/router/router.go`、`internal/handler/{knowledge,faq,evaluation,wiki_page,datasource,tag}.go`
- 问答管线：`internal/application/service/chat_pipeline/{query_understand,query_expansion,search,search_entity,rerank,merge,into_chat_message,chat_completion_stream,load_history,wiki_boost}.go`
- 入库：`internal/application/service/{knowledge_process,knowledge_create,image_multimodal}.go`、`internal/types/{task,task_dead_letter}.go`
- 解析/分块：`docreader/parser/{registry,chain_parser,pdf_parser,concurrency,xlsx_repair}.py`、`docreader/splitter/{splitter,header_hook}.py`、`internal/infrastructure/chunker/{heading_splitter,splitter,header_tracker}.go`
- 图谱：`internal/application/service/graph.go`

**KnowLink 侧**：
- 问答：`src/knowlink/api/routers/{chat,retrieval}.py`、`retrieval/{hybrid,fusion,rerank_stage,graph_boost,faq}.py`、`generation/{answer,context,citations}.py`、`llm/rerank.py`
- 入库：`parsing/{registry,pdf_parser,docx_parser}.py`、`chunking/{__init__,heading,parent_child}.py`、`indexing/{pipeline,embedder,image_pipeline,reconcile}.py`、`tasks/parse_tasks.py`
- 图谱/Wiki：`graph/builder.py`、`wiki/`
