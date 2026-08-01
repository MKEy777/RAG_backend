# 前后端修改记录

本文档记录 RAG 项目前后端一系列 Bug 修复与交互优化的变更。

---

## 1. 聊天页历史会话重复

**问题**:`/chat` 页面「历史会话」列表与「最近」列表重复展示。

**文件**:`RAG_frontend/src/views/Chat/ChatView.vue`

**修改**:删除重复的「历史会话」侧栏及顶部菜单按钮，只保留「最近」；同步清理相关脚本、import 与样式。

---

## 2. 检索测试页字数计数重复

**问题**:`/retrieval` 页面查询框出现两个 `0/500` 计数。

**文件**:`RAG_frontend/src/views/Retrieval/TestView.vue`

**根因**:手动 `span` 计数 + TDesign 组件自带 `maxlength` 计数叠加。

**修改**:删除手动 `0/500` 计数及 `.query-count` 样式，依赖组件自带计数。

---

## 3. 新建知识库页字数计数重复 + 创建 422 报错

**文件**:
- `RAG_frontend/src/views/KnowledgeBase/CreateView.vue`
- `RAG_backend/src/knowlink/settings.py`
- `RAG_backend/src/knowlink/api/routers/knowledge_bases.py`
- `RAG_backend/scripts/e2e.py`

**问题一(计数重复)**:`0/400` 出现两次。
**修改**:删除手动计数，为输入框加 `show-limit-number`,使用组件自带计数。

**问题二(创建报 422)**:前端默认 `embedding_model_id` 与后端 `settings.embedding_models` 的 key 不匹配。
**根因与决策**:统一以完整模型名 `Qwen/Qwen3-Embedding-0.6B` 作为 id。
**修改**:
- `settings.py`:`embedding_models` 的 key 改为 `Qwen/Qwen3-Embedding-0.6B`
- `knowledge_bases.py`:`KBCreate` 默认值同步为完整模型名
- `e2e.py`:`embedding_model_id` 同步修正
- `CreateView.vue`:修正默认值，并在加载后自动校正为合法 id

---

## 4. 知识库详情页重复上传入口

**问题**:`/knowledge-bases/{id}` 顶部有「导入文件」按钮，下方又有拖拽上传区,重复。

**文件**:`RAG_frontend/src/views/KnowledgeBase/DetailView.vue`

**修改**:删除顶部「导入文件」按钮(拖拽区本身支持点击上传)。

---

## 5. 知识 Wiki 入口交互优化

**问题**:详情页 Wiki 标签页点进去只是一个「进入 Wiki 界面」的占位跳转按钮。

**文件**:`RAG_frontend/src/views/KnowledgeBase/DetailView.vue`

**修改**:删除 Wiki 标签页,在顶部「知识图谱」按钮旁新增「知识 Wiki」按钮(`v-if="kb?.enable_wiki"`),点击直接跳转 `/knowledge-bases/{id}/wiki`;清理 tabs 结构、`activeTab` 及相关样式,文件列表直接展示。

---

## 6. AI 图标功能说明

**位置**:详情页顶部「AI 图标」按钮。

**逻辑**:
- 前端 `handleSuggestIcon`:调用 `suggestIcon(kbId)` → `updateKnowledgeBase` 写入 icon → 刷新 kb → 提示"已设置图标"
- 后端 `suggest_icon`(`POST /{kb_id}/icon/suggest`):用知识库名称+描述构造 prompt,调 LLM 让其挑选最贴切的 emoji,取首个 token 截断,失败默认 📚,存库返回

**作用**:让大模型根据知识库名称与描述推荐一个 emoji 作为图标,用于列表/展示的视觉标识。

---

## 7. 连续上传第二个文件报 database is locked

**文件**:`RAG_backend/src/knowlink/indexing/pipeline.py`

**根因**:后台索引管线先 `session.flush()` 写入 chunks(开启 SQLite 写事务并持锁),随后 `await embed_texts()` 调用嵌入模型(耗时数十秒),期间写锁一直不释放。第二个文件上传的 INSERT commit 抢不到锁,超过 `busy_timeout`(30s)后失败。

**修改**:
1. `_index_parent_child` / `_index_flat`:将嵌入前的 `session.flush()` 改为 `session.commit()`,先提交 chunks 释放写锁,再做耗时嵌入,使并发上传能正常写库。
2. 异常分支补偿:因 chunks 已提前提交,嵌入/向量写入失败时新增软删 `new_ids` 对应 chunks 的逻辑,保证数据一致性。

**验证**:重启后端后连续上传多个文件不再互相锁死。

---

## 8. .doc 文件解析出乱码(未走 LibreOffice 转换)

**文件**:`RAG_backend/src/knowlink/parsing/registry.py`

**问题**:上传老版 `.doc`(OLE 复合文档)时,日志显示 `format=ole`、「未找到对应解析器_使用纯文本回退」,导致把二进制内容当纯文本索引出乱码。

**根因**:`detect_format` 对 OLE 文件的后缀映射遗漏了 `.doc` 与 `.ppt`,只判断了 `.docx`/`.pptx`/`.xlsx`/`.xls`。真正的 `.doc` 文件落到 `return "ole"`,而 `_PARSERS` 中并无 `ole` 解析器,回退纯文本。

**修改**:补全 OLE 后缀映射,`.doc`/`.docx` → `doc`,`.ppt`/`.pptx` → `ppt`,`.xls`/`.xlsx` → `xls`,使 `.doc` 正确路由到基于 LibreOffice 转换的 doc 解析器(系统已安装 soffice)。

---

## 9. Wiki 无对照库/概念库/实体类型/领域综览库(分类页面缺失)

**文件**:
- `RAG_backend/src/knowlink/wiki/ingest.py`
- `RAG_backend/src/knowlink/wiki/finalize.py`
- `RAG_backend/src/knowlink/wiki/dedup.py`

**问题**:Wiki 构建后左侧只有「概览/目录」,缺少概念库、实体类型、领域综览库、对照库分组,且目录点不开。

**根因**:所用 `deepseek-v4-flash` 为推理模型,其 `reasoning_content` 与正文 `content` 共享 `max_tokens` 额度。抽取类调用(`extract_candidates` 等)原额度过小(4096),推理过程占满后正文 JSON 被截断(`finish_reason=length`),`content` 返回空 → 候选实体/概念解析为 0 → 无 concept/entity 页面 → 左侧对应分组不显示。静默 `except: return []` 掩盖了截断错误。

**修改**:
1. 全面调大 JSON 抽取类调用的 `max_tokens`,为 reasoning + 正文留足额度:
   - `ingest.extract_candidates` 4096 → 16384(核心)
   - `ingest.citation` 2048 → 8192
   - `dedup.dedup` 2048 → 8192
   - `finalize.classify`(taxonomy) 2048 → 8192
   - `finalize.generate_intro` 1024 → 4096
2. `finalize._rebuild_index_page`:修复被误删的 `index_page = WikiPage(` 结构,补 fallback 兜底(intro 为空/异常时回退目录页),避免 index 空内容。

**验证**:同一文档候选数由 0 提升到 27,后端语法检查通过。重建 Wiki 后概念库/实体类型等分组正常显示。

**补充**:
- 「更新 Wiki」调 `run_wiki_build(enqueue_missing=True)`,引导扫描用 `~processed_op` 排除已完成文档,故已 build-completed 的文档不会重跑抽取——旧库需重置 `wiki_pending_operations` 的 build op(completed_at 置空)才会应用新 token 生效。手动重跑后实测 concept 32 / entity 21。
- `finalize`:`generate_comparisons`(对照库)依赖页面 `out_links` 共现,但 `_inject_cross_links` 原在其后执行,导致 comparison 运行时 out_links 为空、无对照页。已将 `_inject_cross_links` 提前到 comparison 之前。
- `synthesis`(领域综览库)为可选功能,需在 `wiki_config.synthesis_model` 配置模型,未配置则不生成。本库已配 `deepseek-v4-flash` 并重跑 finalize,生成领域综览页 1 篇。
- `comparison`(对照库)需实体/概念页在 `out_links` 中共现 ≥2 次才生成;本库文档间实体交叉不足,暂无对照页,属数据特性而非缺陷。

**说明**:此前上传产生乱码的 `.doc` 文档建议删除后重新上传。

---

## 10. Wiki 关系图谱缩放修复

**问题**:Wiki 侧栏关系图谱缩放/平移功能不稳定,常出现只能缩小无法放大。

**文件**:`RAG_frontend/src/views/Wiki/WikiView.vue`

**修改**:
1. `renderGraph` 改为每次 `dispose` 后重建 chart(避免 setOption 复用导致的状态异常)。
2. 图 series 增加 `scaleLimit: { min: 0.4, max: 8 }`,明确允许放大到 8 倍。
3. 末尾补 `nextTick(() => chart.resize())`,修复侧栏容器初次布局尺寸为 0 导致 roam 缩放失效。

**说明**:节点点击仍为选中对应 Wiki 页(关系图谱与知识图谱为独立功能,不做跳转)。

**分类谱系**:侧栏关系图谱固定按当前选中页所属分类(kind)显示该分类下全部节点及其内部关系(`displayGraph` 按 `selected.kind` 过滤)。标题旁「查看全量图谱」按钮改为弹出 `t-dialog` 大窗口(80% 宽 / 70vh 高),内部独立 `fullChart` 渲染全量 `graphData`(节点更大、名称更长);点击弹窗节点选中对应页并关闭弹窗,关闭时 dispose 释放。

## 11. Wiki 领域综览库支持多主题页

**问题**:领域综览库每个知识库只生成一个笼统的「领域综览」页,无法按子领域/业务线拆分展示。

**文件**:`RAG_backend/config/prompts/wiki.yaml`、`RAG_backend/src/knowlink/wiki/synthesis.py`

**修改**:
1. `WikiSynthesisPrompt` 改为要求 LLM 按内容真实存在的跨文档主题/子领域拆分(有几个生成几个,单一主题只出 1 页,不硬凑数量),输出 JSON `{"pages":[{title,summary,content}]}`。
2. `generate_synthesis` 新增 `_parse_synthesis_pages`(解析 JSON,兼容 ```json 包裹);生成前软删除旧 synthesis 页,再按主题逐条 `_upsert_page`(slug=`synthesis/topic-{i}`,标题取 LLM 返回)。
3. `max_tokens` 由 4096 提升至 8192 以容纳多页内容。

**说明**:前端侧栏与 index 接口按 kind 分组,自动列出多个 synthesis 页,无需改前端。需重新生成 Wiki 才会呈现多页效果。

## 12. Wiki 构建进度反馈

**问题**:点击「更新 Wiki」后无法得知是否成功、当前界面是新是旧、以及构建进度。

**文件**:`RAG_backend/src/knowlink/api/routers/wiki.py`、`RAG_frontend/src/api/wiki.ts`、`RAG_frontend/src/views/Wiki/WikiView.vue`

**修改**:
1. 后端 `wiki_status` 返回值扩展:`total_operations`(总任务数)、`completed_operations`(已完成数=总数-pending)、`building`(pending>0)、`last_updated`(`func.max(WikiPage.updated_at)`,ISO 字符串)。
2. 前端 `WikiStatus` 接口同步新增上述字段。
3. `WikiView.vue`:侧栏 status-bar 构建中显示进度条(`t-progress`,基于 completed/total 百分比)+「构建中 X/Y」标签,非构建态显示「最近更新:格式化时间」;`fetchStatus` 用 `building` 字段判断完成跃迁(prev building→false)并 `MessagePlugin.success('Wiki 更新完成')`;新增 `formatTime` 格式化 ISO 时间。

## 13. 弹窗高度超出视口需滚动修复

**问题**:知识库「设置」弹窗与 Wiki「查看全量图谱」弹窗内容高度超过视口,底部被截断,需要滚动才能看全。

**文件**:`RAG_frontend/src/views/KnowledgeBase/DetailView.vue`、`RAG_frontend/src/views/Wiki/WikiView.vue`

**修改**:
1. 两个 `t-dialog` 均加 `top="5vh"` 定位到顶部,并加专属 class。
2. `:deep(.t-dialog)` 限制 `max-height: 90vh` + flex 纵向布局,`:deep(.t-dialog__body)` 设 `flex:1; min-height:0`;设置弹窗 body `overflow-y:auto`(表单滚动),全量图谱弹窗 body `overflow:hidden`(图谱固定 70vh 自适应)。

## 14. 聊天参考来源仅保留实际引用 + 修复点击 404

**问题**:①聊天回答把检索到的全部 topk 候选都列为参考来源,不准确;②点击「未命名片段」(wiki/faq 类候选,无有效 document_id)跳转 `getDocument('')` 报 404。

**文件**:`RAG_backend/src/knowlink/generation/citations.py`、`RAG_backend/src/knowlink/generation/answer.py`、`RAG_frontend/src/components/chat/ChatReferences.vue`

**修改**:
1. `citations.py` 新增 `extract_cited_ids`(正则解析答案里的 `<ref id="cN"/>`);`build_references` 增加可选 `cited_ids` 参数,仅保留被引用的候选。
2. `answer.py` 流式与非流式版均改为:生成完成后用完整答案 `extract_cited_ids` 过滤 references;若答案未含任何 `<ref>`(cited_ids 为空)则回退列出全部,避免有答案却无来源。
3. 前端 `ChatReferences.vue`:`openChunk` 对无 `document_id` 的候选直接 return 不跳转;无 document 项加 `is-disabled` 样式(默认光标、无 hover);空 `title_path` 时按 channel 显示友好标题(FAQ 问答 / Wiki 页面 / 文档片段等)替代「未命名片段」。

---

## 15. 知识库设置由弹窗改为独立页面

**问题**:知识库「设置」使用 t-dialog 弹窗，内容较多导致底部被截断、体验差。

**修改**:
1. 新增 `RAG_frontend/src/views/KnowledgeBase/SettingsView.vue`:整页表单,顶部返回箭头 +「设置知识库」标题;卡片式基本信息(名称/描述/Wiki/知识图谱)与检索设置(复用 SearchSettingsForm);底部取消/确认按钮;进入按 :id 拉取详情填充。
2. `RAG_frontend/src/router/index.ts` 新增路由 `knowledge-bases/:id/settings`。
3. `RAG_frontend/src/views/KnowledgeBase/DetailView.vue`:`openSettings` 改为 `router.push` 跳转;移除设置 t-dialog 模板、settingsVisible/savingSettings/settingsForm/searchSettings 状态、saveSettings 函数及 settings-dialog 等样式与 SearchSettingsForm import。

---

## 16. 关系图谱按文档高亮

**问题**:从某文档进入知识图谱时,该文档抽取出的实体与关系无法与其它文档区分。

**文件**:
- `RAG_backend/src/knowlink/api/routers/graph.py`
- `RAG_frontend/src/api/graph.ts`
- `RAG_frontend/src/views/Graph/GraphView.vue`
- `RAG_frontend/src/components/DocDetailDrawer.vue`

**修改**:
1. 后端 `get_graph` 新增可选 `knowledge_id` 参数:据此查该文档(knowledge_base_id+knowledge_id+未删)的 chunk 集合,节点/边的 `source_chunk_ids` 与之相交则 `belongs=True`,否则 `False`;`GraphNode/GraphEdge` 新增 `belongs` 字段(默认 True,向后兼容)。
2. 前端 `api/graph.ts`:`GraphNode/GraphEdge` 加 `belongs?: boolean`;`getGraph` 加 `knowledgeId?` 参数透传 `knowledge_id`。
3. `GraphView.vue`:读 `route.query.knowledge_id` 传参;`renderChart` 对 `belongs===false` 的节点降低 opacity 并隐藏 label、对相应边降低 lineStyle.opacity 淡化;顶部 alert 提示「仅高亮当前文档」并提供「显示全部」清除入口。
4. `DocDetailDrawer.vue`:基本信息页在图谱已完成时新增「在图谱中查看本文档实体」按钮,跳转 `graph?knowledge_id={doc.id}`。
5. `RAG_frontend/src/views/Wiki/WikiView.vue`:Wiki 页内嵌关系图谱原本全部同色无法分辨当前概念。改为 `renderGraph` 依据当前选中页 slug 计算邻接集,当前节点放大+红色描边+加粗,直接相连节点保持高亮,其余节点/边淡化(opacity);`highlightGraph` 改为重渲染触发。

---

## 17. 聊天参考来源切片跳转:返回路径与切片定位

**问题**:
1. 点击聊天参考来源的切片跳转到切片详情页后,点返回按钮走到了知识库界面而非返回聊天页。
2. 点击切片只跳到源文最开始,未定位到对应的具体切片。

**文件**:
- `RAG_frontend/src/components/chat/ChatReferences.vue`
- `RAG_frontend/src/components/chat/CitationPopover.vue`
- `RAG_frontend/src/views/Documents/ChunksView.vue`

**修改**:
1. 两处跳转入口 `openChunk` 改为携带 query `{ from: 'chat', chunk_id }`。
2. `ChunksView.vue` 返回按钮 `back-to` 改为动态 `backTo`:`from=chat` 时返回 `/chat`,否则原知识库详情页。
3. `ChunksView.vue` 定位模式(存在 `chunk_id`):类型筛选初始置空(全部类型),新增 `locateTargetChunk` 逐页(size=100)查找目标切片,命中后设置分页并 `selectChunk` 高亮源文 + 滚动切块列表到该项;`loadDoc` 在定位模式下跳过默认 `fetchChunks` 以避免竞态。

---

## 18. 后端日志时间与级别格式优化

**文件**:`RAG_backend/src/knowlink/logging.py`

**修改**:
1. 新增 `TimeStamper`,为每条结构化日志补充 `HH:MM:SS` 时间戳。
2. 新增 `_time_level_renderer`,将日志输出统一渲染为包含级别、时间、事件及上下文字段的单行格式,并保留级别颜色。

**作用**:让后端控制台日志更容易按时间和级别定位问题,同时保留请求追踪等上下文字段。

---

## 19. 检索设置表单 `v-model` 响应式修复

**文件**:`RAG_frontend/src/components/SearchSettingsForm.vue`

**问题**:组件将 `props.modelValue` 直接保存为普通常量;父组件替换整个设置对象后,表单更新函数可能继续基于旧对象合并字段。

**修改**:将 `modelValue` 包装为 `computed`,各更新入口统一从 `model.value` 读取当前值再发出 `update:modelValue`。

**作用**:保证知识库独立设置页等复用场景中,检索设置始终基于最新的父级状态更新。

---

## 20. 移除独立 FAQ 知识库类型(FAQ 归并为 wiki 副产物)

**背景**:FAQ 切片已在 wiki 构建流程中由 `generate_faq_chunks` 自动生成(`chunk_type=="faq"`),且 `hybrid_search` 不按 chunk_type 过滤,普通检索本就能召回 FAQ 切片。独立 FAQ 库类型与旁路检索属冗余设计,予以移除。

**后端**:
1. `retrieval/pipeline.py`:删除 `faq_search` import 与 `kb.type=="faq"` 分流,所有库统一走 `hybrid_search` + RRF 融合 + graph/wiki 加权;删除 `RetrievalResult.faq_hits` 与 `enrich_faq_candidates`。
2. `api/routers/retrieval.py`:删除 faq_hits 构建,`RetrievalResponse` 仅返回 `hits`。
3. `api/routers/chat.py`:candidates 直接取 `result.hits`,去除 faq 拼接。
4. `api/app.py`:移除 faq 路由注册。
5. `retrieval/faq.py`、`api/routers/faq.py`:清空为占位说明。

**前端**:
1. `KnowledgeBase/CreateView.vue`:删除知识库"类型"表单项(FAQ 选项)。
2. `KnowledgeBase/ListView.vue`:删除类型筛选与卡片类型标签。
3. `router/index.ts`、`layouts/MainLayout.vue`:删除 FAQ 管理路由与菜单。
4. `api/faq.ts`、`views/Faq/FaqView.vue`:清空为占位。
5. `api/retrieval.ts`、`views/Retrieval/TestView.vue`:删除 `faq_hits` 类型字段及展示/样式。

**保留**:数据库 `type`/`faq_config` 列保留(删列迁移风险大),仅移除逻辑分流;FAQ 自动生成机制(`build.py` 调 `generate_faq_chunks`)完整保留。

**作用**:统一检索管线,FAQ 作为 wiki 副产物切片走普通 hybrid_search,消除冗余旁路与独立库类型。

---

## 21. Wiki 页面展示自动生成的 FAQ 问答

**背景**:FAQ 独立管理入口移除后,wiki 自动生成的 `chunk_type=="faq"` 问句切片缺少可视化入口。

**后端**:`api/routers/wiki.py` 新增 `GET /wiki/{kb_id}/faq`,查询 `chunk_type=="faq"` 切片并按来源文档(`knowledge_id`→`Knowledge.title`)分组返回问句列表。

**前端**:
1. `api/wiki.ts`:新增 `listWikiFaq` 及 `WikiFaqGroup`/`WikiFaqQuestion` 类型。
2. `views/Wiki/WikiView.vue`:左侧导航新增可折叠「FAQ 问答」分组(按文档列条目);点击文档后中间正文以 `faq-list` 展示该文档全部自动生成的 FAQ 问句;`fetchPages` 并发拉取 FAQ,`selectSlug` 识别 `faq:` 前缀切换 FAQ 视图。

**作用**:让用户在 Wiki 页面直接查看各文档自动生成的 FAQ 问句,补齐可视化入口。

**补充清理**:移除 `ListView.vue` 遗留的 `typeFilters`(含 FAQ 选项,已无引用)与 `DetailView.vue` 的文档/FAQ 类型标签。

## 22. 侧边栏新增「知识 Wiki」「知识图谱」全局入口

**背景**:Wiki/图谱路由均需指定 `:id`,缺少全局入口,只能从知识库详情进入。

**前端**:
1. `views/KnowledgeBase/SelectKbView.vue`(新增):全局选择页,按 `route.meta.mode`(wiki/graph)展示对应可用知识库卡片,点击跳转 `/knowledge-bases/:id/{mode}`。
2. `router/index.ts`:新增 `/wiki`、`/graph` 路由,复用 `SelectKbView`,以 `meta.mode` 区分。
3. `layouts/MainLayout.vue`:侧边栏「知识库」下方新增缩进子项「知识 Wiki」「知识图谱」,收起态取消缩进。

**作用**:提供全局 Wiki/图谱入口,先选知识库再查看对应内容。

---

## 23. 快路径（fastpath）三层路由 + 语义缓存 + 管理界面

**背景**:高频重复问题每次仍走完整检索+生成链路(耗时数秒)。为常见问答建立"先命中、后生成"的快路径,命中时毫秒级返回。

**架构**:聊天入口在 rewrite 之后、retrieve_candidates 之前插入三层路由(见 `docs/ARCHITECTURE.md` §4.9):
- L0 快路径命中(~50ms-1s):`fastpath/matcher.py` 归一化精确匹配(去空白/标点/NFKC/小写,零嵌入开销)→ 向量语义匹配(阈值 `fastpath_match_threshold=0.88`);命中条目含 `{var}` 时 `fastpath/filler.py` 用目标 chunk 动态取数填充模板(任一变量缺失→落 L2 不写死答案)。
- L1 语义缓存(~150-300ms):`fastpath/cache.py`,`semantic_cache_entries` 表 + `sc_{kb_key}` 向量命名空间,写穿缓存有引用且源 chunk 存活的答案;命中校验全部源 chunk `content_hash` 未变,任一变更即失效;TTL 7d、每库上限 500 条 LRU。
- L2 现有完整链路(不变),成功后后台写穿缓存(`write_cache_background` 独立 session,不阻塞 SSE)。

**后端**:
1. `migrations/versions/0006_fastpath.py`(新增):`fast_path_entries`、`semantic_cache_entries`、`fast_path_hits` 三张表。
2. `models/fastpath.py`(新增):`FastPathEntry`/`SemanticCacheEntry`/`FastPathHit` 模型;`models/__init__.py` 注册。
3. `fastpath/`(新增):`matcher.py`/`filler.py`/`cache.py`/`mining.py`/`service.py`。`mining.py` 离线挖掘:`chat_messages`+`retrieval_tests` 问句贪心聚类(阈值 0.85,簇≥3)→ 跑完整检索定位 → LLM 生成模板 → `status=draft` 待审核。
4. `api/routers/fastpath.py`(新增):条目 CRUD/启停/激活(激活时嵌问法向量到 `fp_{kb_id}`)、FAQ 候选列表 + 一键升级(`POST /{kb_id}/promote`)、挖掘(`POST /{kb_id}/mine`)、命中统计、缓存列表/清除。
5. `api/app.py`:注册 fastpath 路由。`api/routers/chat.py`:chat 快路径命中时新增 SSE `via` 事件(`{"via":"fastpath"|"cache"}`)前置。
6. `retrieval/pipeline.py`、`api/routers/retrieval.py`:fastpath 编排接入与历史命中记录。
7. `settings.py`:新增 `fastpath_match_threshold`/`fastpath_fill_max_tokens`/`cache_ttl_s`/`cache_max_entries_per_kb`/`mining_*` 等配置。
8. `storage/vector_local.py`:`fp_{kb_id}`/`sc_{kb_key}` 命名空间支持。`metrics.py`:`knowlink_fastpath_hits_total{layer}` + `knowlink_fastpath_duration_seconds{layer}`。
9. `api/routers/documents.py`:reparse/delete 文档时 `mark_entries_stale_for_document`/`invalidate_cache_for_chunks` 使相关快路径条目与缓存失效。
10. `config/prompts/fastpath.yaml`(新增):变量抽取(`variable_extraction`)与模板生成(`template_generation`)提示词。
11. `tests/test_fastpath_*.py`(新增 6 个):matcher/cache/mining/fill/API/chat 集成测试。

**前端**:
1. `api/fastpath.ts`(新增):`FastPathEntry`/`FaqCandidate` 等类型及条目/候选/挖掘/统计/缓存接口。
2. `views/FastPath/FastPathView.vue`(新增):快路径条目管理界面(列表/启停/激活/升级/挖掘/缓存清除/命中看板)。
3. `api/chat.ts`、`composables/useChatStream.ts`、`stores/chat.ts`、`components/chat/ChatMessage.vue`:解析 `via` 事件,快路径命中时聊天消息显示命中标记。
4. `router/index.ts`、`layouts/MainLayout.vue`:新增「快路径」管理路由与侧边栏入口。

**验证**:聊天/检索高频问题命中后返回毫秒级;写穿缓存源 chunk 变更后自动失效重生成。

---

## 24. 检索测试历史记录显示 P50/P99 延迟分位数

**背景**:检索测试页「测试历史」只展示总耗时与命中数,无法判断单次检索中候选排名的延迟分布,排查慢请求困难。

**后端**:
1. `models/retrieval_test.py`:`RetrievalTest` 新增 `latency_p50_ms`/`latency_p99_ms`(Float, nullable)。
2. `migrations/versions/0007_retrieval_test_latency.py`(新增):`retrieval_tests` 表加两列。
3. `api/routers/retrieval.py`:`_save_test_history` 写入当次检索后按 `latency_samples` 计算的滚动 P50/P99(毫秒,四舍五入到 1 位小数);`GET /retrieval/history` 列表返回 `latency_p50_ms`/`latency_p99_ms`。

**前端**:
1. `api/retrieval.ts`:`RetrievalHistoryItem` 新增 `latency_p50_ms?`/`latency_p99_ms?`。
2. `views/Retrieval/TestView.vue`:历史记录行内显示 `P50 xx ms · P99 xx ms`(旧记录无分位数自动省略)。

**作用**:每条检索历史可直接看到 P50/P99 延迟分位,便于定位检索管线瓶颈。
