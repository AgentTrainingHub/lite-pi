# 源码入口

`lite_pi.session.AgentSession` 是公开会话 API，`lite_pi.__main__` 是 CLI。两者共用 `loop`、`provider` 和 `tools.registry`，请求由 `spend` 统一计费。

`journal` 管理持久历史及分支；`storage` 提供锁与原子文件写入。`tools.workspace` 管理文件边界，`tools.shell_workspace` 管理过滤项目副本及效果发布，`tools.shell` 管理容器生命周期。模块依赖、支持范围和失效语义见[架构说明](../ARCHITECTURE.md)。
