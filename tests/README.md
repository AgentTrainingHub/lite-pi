# 测试与验证

确定性机制测试使用标准库 unittest，验证工具协议、权限、预算、持久化、恢复及终端交互。允许构造工具调用、模型响应和 HTTP 输入；这些结果标记为 `deterministic`。MCP、SQLite、文件、浏览器和容器的效果按用例实际运行，机制测试不调用付费模型。

端到端测试从公开 `AgentSession` 或 CLI 进入，连接真实 DeepSeek，标记为 `real_model`。模型回复由实际请求产生，原始响应、用量、工具效果和失败尝试分别保存。效果判断使用独立的文件、进程、数据库、服务状态与请求回执。

## 运行入口

在项目根目录安装依赖后执行免费的机制测试：

```bash
uv sync --locked --group compare
uv run --group compare python tests/contracts.py
```

需要 Python 3.12 和 compare 依赖。浏览器用例需要 Playwright Chrome；Shell 用例需要运行中的 Docker 和准备后的镜像：

```bash
uv run playwright install chrome
uv run python scripts/prepare_runtime.py
```

环境缺失时，`AcceptanceBlocked` 进入汇总的 `blocked` 列表，默认返回非零。`--allow-blocked` 可用于检查环境不完整时的其余机制，报告仍保留 blocked。CI 使用独立状态目录，不安装凭据，机制测试出现 blocked 即失败。

下面的端到端命令会调用真实模型并产生费用，需要先配置凭据与同一费用账本：

```bash
uv run python tests/acceptance.py tools
uv run --group compare python tests/acceptance.py --help
```

| 分组 | 覆盖 |
| --- | --- |
| `session` | 会话与 CLI、中断与未知用量 |
| `tools` | 工具契约、多轮读取、长行续读、参数错误、轮次限制和错误恢复 |
| `unknown-tool-probe` | 可选未知工具探针；未命中保留 not-covered，独立于必验工具契约 |
| `tool-recovery` | 实际文件错误回填，模型随后读取有效文件并完成任务 |
| `workspace` | 代码修复、功能增加、文件边界及 Shell 隔离 |
| `coding` / `edit` / `shell` | 工作区测试的独立子集 |
| `journal` | 新进程续开、分支投影、文件效果及损坏日志拒绝 |
| `interaction` | 增量流、续开、取消、写入范围、steering 与 follow-up |
| `context` | 摘要与重开、项目指令、Skill；包含 context-boundaries |
| `context-boundaries` | 摘要调用预算、收到响应后取消、指令与 Skill 的权限边界 |
| `extensions` | 扩展工具、双 MCP、资源分页、取消、断线和可执行配置保护 |
| `recovery` | 意图、效果和结果窗口恢复、审批、共享预算；包含 recovery-boundaries 与 concurrency |
| `recovery-boundaries` | 审批崩溃、未知效果、范围恢复、预算和迟到工作者 |
| `concurrency` | 两个只读进程占用额度、其余会话排队与撤销 |
| `knowledge` | 引用改代码、冲突后补查、资料缺失、过期索引和记忆治理 |
| `browser` | 订货、旧观察、响应丢失、状态查询与未知效果 |
| `delegation` | 只读子调查、父任务核验、取消、共享预算和单 Agent 对照 |
| `comparison` | LangGraph、上游比较、保留任务、组合效果及平台检查 |
| `terminal` | 正式 CLI 的 PTY 多轮交互、续开、取消、审批及 MCP 收尾 |
| `terminal-reading` | Markdown、阅读锚点、缩放、草稿、工具详情与响应性 |
| `identity` | 空项目、冲突指令、续开和摘要后的产品、模型及工具身份 |
| `planning` | 计划、要求变化、持久澄清、输入证据及当前审批 |
| `hybrid-retrieval` | BM25、真实 E5、混合检索、重排和来源生命周期 |
| `regressions` / `container` | 文件、流、请求、子进程、容器和效果同步边界 |
| `all` | 调度器中的全部回归，包含收费调用和受保护的保留任务 |

`baselines/coding/heldout/` 只用于正式综合比较，开发与调试不使用该题试错。`all` 和 `comparison` 都会进入保留任务；执行前须确认测试范围与费用授权。

模型能力探测和上游比较使用：

```bash
uv run python scripts/probe_model_capabilities.py
uv run python scripts/compare_upstream.py
```

上游比较还需要固定 Pi 源码及 Node，配置与判定见[比较协议](../baselines/coding/protocol.md)。单独的连通性探测只能证明接口可达。

## 判定与证据

每次机制测试创建 `.artifacts/contracts/<uuid>/`，保存测试身份、日志和 summary。每次端到端测试创建 `.artifacts/acceptance/<group>-<uuid>/`，保存项目、输入输出、diff、独立检查及全部失败。产品回执和真实请求分别位于运行状态目录的 `runs/`、`spend/`；配置见根目录 README。

效果任务独立运行 3 次，至少 2 次满足完整条件；所有尝试的安全断言都必须通过。必验边界须在指定层级命中。未命中为 `not-covered`，环境或额度不可用为 `blocked`，汇总保留 `incomplete` 并返回非零。确定性拒绝、真实模型恢复和模型主动探针分别统计。

独立代码检查在只读项目副本中运行，检查文件在模型运行前固定。Shell 容器清理由独立 Docker 查询判断，容器 PID 不用于宿主信号。凭据检查使用无敏感 canary。未知用量单列并保守计量，部分响应保留原文。

## 代码组织

`contracts.py` 自动发现 `test_*contracts.py` 中本模块定义的 TestCase。测试按 interaction、context、extensions、recovery、knowledge、browser、delegation、graph、terminal、planning 等能力命名；共用夹具放在 `fixtures/`，不通过导入另一个 TestCase 复用。

- `deterministic_transport.py` 控制 HTTP 读取边界，使用独立临时账本，验证真实 Provider 与费用入口的协作。
- `crash_injection.py` 仅在测试子进程显式安装故障点。`retrieval_crash_driver.py`、`browser_crash_driver.py`、`delegation_crash_driver.py` 触发对应的意图窗口中断。
- `fixtures/mcp_server.py` 提供本地协议故障；`fixtures/shop.py` 提供业务状态及请求计数；测试配置不进入产品。
- `fixtures/retrieval/` 保存检索语料；`fixtures/evaluation/` 保存配对任务与受控参照样例。参照样例验证 human-only、mixed、标签隔离和来源统计，不能计为人工校准或模型一致率。
- `compare/` 提供 LangGraph、上游执行臂、配对统计、Judge 和 RAG 诊断。真实 Judge 需要另行提供实际确认的参照与授权记录。
- `cases_integrated.py`、`cases_capability_regressions.py` 和 `integrated_checks.py` 提供组合任务及独立核验，可从对应模块入口运行。

## 终端检查

```bash
PYTHONPATH=tests uv run python -m unittest test_terminal_contracts test_terminal_reading_contracts -v
uv run --group compare python tests/acceptance.py terminal
uv run --group compare python tests/acceptance.py terminal-reading
```

机制测试使用真实 PTY 进程、键盘输入、ANSI 文本和 termios 检查；`fixtures/tui_driver.py` 的受控响应只用于这一层，前后核对主账本请求数不变。`tui_pty.py` 负责释放描述符和回收子进程，pyte 的宽字符占位符按空白单元格呈现。阅读测试覆盖 1000 条历史、64 KiB 活动正文、50 事件/秒、大回执展开、折叠锚点和未提交草稿。

真实终端测试从正式 CLI 输入代码任务、转向、follow-up、取消、审批及退出续开；流取消必须先观察到非空真实增量。PTY 文本、原生桌面视觉、容器和独立宿主分别记录。

静态检查另行执行：

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy
```

格式、类型、链接和静态检查只证明对应项目通过，不计作功能、真实模型或平台验收。
