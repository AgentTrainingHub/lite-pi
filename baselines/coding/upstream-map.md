# 上游源码与行为依据

lite-pi 参照 Pi commit `7d75a0e3eb273dff753d49c066ea29fbe7b3b6f6`，使用 Python 3.12 实现会话、工具执行、历史压缩与宿主约束。相关行为保留在公开接口中，Python 的费用账本、文件边界和持久任务管理由本项目负责。

| Pi 源码 | 参考行为 | Python 实现 |
| --- | --- | --- |
| `packages/coding-agent/src/core/sdk.ts` 的 `createAgentSession` | 组装项目、模型、资源和会话 | `lite_pi.session.AgentSession` |
| `packages/agent/src/agent-loop.ts` | 模型轮次、工具结果回填和下一次请求准备 | `lite_pi.loop`、`lite_pi.control` |
| `packages/coding-agent/src/core/compaction/compaction.ts` | 完整消息组切点、轮内压缩和摘要 | `lite_pi.compaction`、`lite_pi.journal` |
| `packages/evals/src/harness.ts` 的 `createPiCodingAgentHarness` | 从完整任务取得实际产物 | `tests/compare/` 与独立效果检查 |
| `packages/evals/src/report.ts` 的 `summarizeEvalObservations` | 保留失败、缺失和不稳定结果 | 逐次回执与完整集合汇总 |
| `packages/coding-agent/package.json` 的 `files` | 明确列出发布包内容 | `pyproject.toml` 的 wheel 与源码包清单 |

上游比较通过固定版本 CLI 和本机费用代理运行，`PI_CODING_AGENT_DIR` 使用独立目录，并关闭上下文文件、MCP 和会话自动发现。模型目录使用 `thinkingFormat: deepseek`；关闭思考后发送 `thinking.type=disabled`，代理同时固定这一请求约束。

Python 实现的支持范围以 [README](../../README.md) 和[架构说明](../../ARCHITECTURE.md)为准。Pi 日志格式、其他 Provider 和 TypeScript 插件需要各自的适配，本项目当前没有提供这些兼容接口。
