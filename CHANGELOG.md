# Changelog

## [Unreleased]

### Added (feature/wiki)
- Wiki 去重归并接线（表面相似度预筛 + LLM 去重判定）
- 删除文档触发 wiki 撤回全链路（含向量/关键词索引清理）
- 重解析 replace 语义（stale 撤回 + 替换指令合并）
- Index 树形导航 API（GET /wiki/{kb_id}/index）
- Synthesis/Comparison 终态自动生成（synthesis_model 配置驱动）
- FAQ：自动问句生成 + 手动导入 API + 独立检索路径（标签优先级）

### Changed (feature/wiki)
- Wiki 提示词全部迁移 config/prompts/wiki.yaml
- 概览页 slug 改 overview/<doc_id>

### Removed (feature/wiki)
- 未使用的 wiki 发布层 10 表（含迁移 4cc06c3b8b09）

### Added
- Project scaffold: initial project structure, architecture document, and coding conventions
