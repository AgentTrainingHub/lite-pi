# lite-pi

参照 Pi 源码实现的 Python 3.12 Coding Agent。支持持续交互终端、精确文件修改、隔离 Shell、会话续开与分支、任务审批和恢复、本地 BM25/语义/混合检索、受治记忆、浏览器工具、有限子 Agent 调查，以及可选计划和持久澄清。测试工具支持框架对照、配对评估、消融与 RAG 诊断。

- [架构与支持范围](ARCHITECTURE.md)
- [测试方法与运行入口](tests/README.md)
- [上游源码与行为依据](baselines/coding/upstream-map.md)

当前验证范围包含本机运行和 Linux 容器；独立干净 macOS、Linux、WSL2 宿主、完整人工 Judge 校准及桌面视觉仍待验证。

## 准备环境

在项目根目录执行。需要 Python 3.12、uv 和运行中的 Docker 引擎。`compare` 组只安装 LangGraph 对照，产品运行不需要它：

```bash
uv sync --locked
uv sync --locked --group compare
mkdir -p ~/.lite-pi
test -e ~/.lite-pi/.env || cp .env.example ~/.lite-pi/.env
chmod 600 ~/.lite-pi/.env
```

在 `~/.lite-pi/.env` 中填写自己的 DeepSeek API key；已有配置时保留原文件。当前模型固定为 `deepseek-flash`，endpoint 为 `https://api.deepseek.com`，关闭思考模式。密钥只在宿主请求进程读取，不进入命令环境。

```bash
uv run python scripts/prepare_runtime.py
uv run python -m lite_pi.spend init
uv run python -m lite_pi.spend status
```

镜像准备会下载官方 `python:3.12-slim`，将不可变摘要存入运行状态目录的 `runtime/image.json`；之后 Shell 只使用该摘要。费用初始化查询官方余额并建立或续用同一账本，不重置已花金额。当前费用上限为 15 元，账户保留额为 10 元。初次建账时，余额扣除保留额后与费用上限取较小值，最低为零；已有账本沿用其中的额度和消费。单次输出上限为 2048 tokens，最多一条在途请求；发送重试最多一次，也消耗调用预算。计量与基线比较方法见[比较协议](baselines/coding/protocol.md)。

Linux 使用 `/proc/<pid>/stat` 的启动时间与当前 boot ID 检查进程身份，无须安装 `procps`。macOS 使用系统 `/bin/ps`。升级遇到仍存活的旧格式 Linux 所有者时保留其并发槽，待原进程退出后回收。在 Docker 中运行主程序时，项目状态、SQLite 和 `.artifacts` 应位于 Linux 原生卷。当前 Docker Desktop 的 macOS 共享目录曾出现 `flock` 未排斥并发写入的实测失败，不适合存放这些可写状态；容器内 Shell 使用同一 Docker 引擎时，工作目录还必须能被该引擎按相同绝对路径挂载。

## 运行数据的位置

安装为 wheel 后也可从任意工作目录运行 `lite-pi --help` 和 `lite-pi chat --root /absolute/project`。运行状态默认位于 `~/.lite-pi`，包含 `spend/`、`runs/`、`locks/`、知识库访问锁及镜像摘要；知识库正文仍位于目标项目 `.lite-pi/knowledge.sqlite`。TUI 的可续接 Journal 仍放在目标项目 `.artifacts/sessions/`，运行回执中的 `evidence` 指向宿主证据目录。

路径在进程启动时统一读取，优先级为环境变量 `LITE_PI_HOME` / `LITE_PI_ENV_FILE`、`~/.config/lite-pi/runtime.toml`、默认值。显式配置的凭据路径独立于状态目录；未配置时为状态目录中的 `.env`。`runtime.toml` 中的相对路径以该文件所在目录为基准，值须为非空字符串，格式错误时启动即报出文件路径。同一宿主的所有 Session、进程、项目锁与费用账本共用这份配置，避免按项目产生多个预算。

需要沿用已有状态目录时，可将下列路径替换为实际绝对路径：

```toml
# ~/.config/lite-pi/runtime.toml
state_dir = "/absolute/path/to/existing-state"
env_file = "/absolute/path/to/existing.env"
```

迁移已有环境时先指向原状态目录；不得新建账本来绕过累计额度。上游比较脚本通过 `LITE_PI_UPSTREAM` 指定 Pi 源码位置，版本见[上游源码与行为依据](baselines/coding/upstream-map.md)。

## 运行

### 持续交互终端

在项目安装目录执行；`--root` 指向要操作的项目：

```bash
uv run lite-pi chat --root /absolute/path/to/project
```

在已经激活的安装环境中，无参数 `lite-pi` 以当前目录进入同一界面。stdin 与 stdout 均须为真实终端，管道输入会返回用法错误。欢迎区显示像素 π、项目、模型、会话和工具授权。输入任务后按 Enter，同一会话会持续接收下一条消息；模型和工具仍由 AgentSession 执行。

| 操作 | 按键或命令 |
| --- | --- |
| 提交 / 运行中转向 | Enter；显式 `/steer 消息` |
| 限定后续写入范围 | `/steer --paths 路径1,路径2 -- 消息` |
| 排队后续任务 | `/follow 消息` |
| 换行 / 粘贴多行 | Alt+Enter、Ctrl+J；粘贴后显式提交 |
| 取消 / 清空草稿 | 运行中 Ctrl+C 或 Esc；空闲 Ctrl+C |
| 回看 / 返回最新 / 展开回执与诊断 | PageUp、PageDown / Ctrl+End / Ctrl+O |
| 会话列表 / 切换 / 新建 | `/sessions`、`/sessions 序号`、`/new` |
| 显式恢复 | `/continue` |
| 审批 | `/approve ID`、`/reject ID`；决定后再 `/continue` |
| 本地帮助 / 状态 | `/help`、`/status` |
| 退出 | `/quit`；空闲且草稿为空时 Ctrl+D |

消息按角色区分，支持标题、强调、列表、行内代码、代码块和基本 diff。回看时新输出保留当前位置，输入草稿保持独立。工具默认显示状态、目标和三行摘要，Ctrl+O 展开回执；再次按下恢复摘要。单个原始证据文件内联读取上限为 64 KiB，超限或不可读会明确提示并保留完整路径，大回执可核对运行证据。设置 `NO_COLOR=1` 使用单色显示，状态仍有文字与符号。

新会话保存在目标项目 `.artifacts/sessions/`；退出显示续开命令。打开已有会话只读取记录，权限沿用持久授权，随后等待输入：

```bash
uv run lite-pi chat --root /absolute/path/to/project --session /absolute/path/to/session.jsonl
```

运行中只能操作当前会话，切换前须等待收尾。未进入请求的排队输入会回填草稿；取消保留已经发生的文件效果。未知效果继续阻止后续调用。审批候选显示工具、参数、对象和预览，批准不会自动执行。`/status` 显示完整费用与任务额度；状态栏区分配置模型、实际模型和未知 token，不显示未经核实的 context 百分比。

界面使用主屏文本布局，支持 80×24、40×12 紧凑模式和 resize。模型、工具、路径和错误中的终端控制字符显示为转义文本。macOS 与 Linux PTY 自动化检查可验证键盘操作和终端恢复；原生桌面视觉与 WSL2 仍待验证。

### 单次任务

下列任务会调用真实模型并产生费用。`--root` 应指向可丢弃的项目副本；会话文件放在该副本之外：

```bash
uv run lite-pi submit --root /absolute/path/to/project '解释 Python 中生成器的执行过程。'
uv run lite-pi code --root /absolute/path/to/project --session /absolute/path/to/session.jsonl --max-turns 8 '先运行 python3 check.py，根据失败修改实现，保留检查文件，最后重新检查。'
uv run lite-pi code --root /absolute/path/to/project --session /absolute/path/to/session.jsonl --delegate --max-task-calls 14 '先做只读调查，再根据宿主回读摘录修改并检查。'
uv run lite-pi resume --session /absolute/path/to/session.jsonl --max-turns 8
uv run lite-pi branch --session /absolute/path/to/session.jsonl --from VALID_RECORD_ID '新的任务要求'
uv run lite-pi resume --session /absolute/path/to/session.jsonl
```

`submit` 无工具，`code` 开放 read/edit/write/bash。模型增量在到达时写到 stderr，同一段最终文本不会再打印一次；工具开始和结束也写到 stderr。stdout 保留一条最终 JSON 回执，内容包括最终文本、模型身份、停止原因、run ID 和证据目录。`submit`、`code`、`resume` 收到 SIGINT 时取消当前运行。CLI 的 `stop`、显式 `interrupt`、取消、`awaiting_approval`、`awaiting_question` 和 `queued` 返回 0；额度受阻、轮次耗尽、任务预算停止和运行失败返回非零；无效参数或损坏会话返回 2。

宿主在每次普通请求中重新装配 lite-pi 身份、配置模型和当前工具清单，续接和压缩后同样适用。项目约定、Skill、记忆和历史分别保留来源，不能扩展工具授权。摘要请求只整理历史，不继续执行任务。模型原始回答保持原文；实际响应模型以回执的 `model` 及 Provider 记录为准。

设置 `--context-tokens` 后，长对话先尝试在用户回合之间压缩；单个回合过长时，在完整的助手调用及工具结果组之间切分，保留该回合的用户原文和最近消息组。原始 Journal 持续保留，摘要及后续回答共用原调用额度。摘要失败显示“摘要失败，历史已保留”；核对原因后可用 `resume` 或 `/continue` 继续，已发生的文件效果不会回滚。轮内压缩记录带 `turn_user_id`，需要支持该字段的当前版本读取。

`code` 和 `resume` 可以加 `--approvals`、`--max-task-calls`、`--max-task-seconds` 和 `--max-task-output`，并可用 `--context-tokens` 限制当次输入与输出预留窗口。审批默认关闭。打开后，写入、命令和扩展工具会先停在等待，回执里的 diff 是预计修改。扩展的 `replay="safe"` 只允许恢复时重试，不免除审批或修改独占。批准后恢复时先持久化执行意图，文件实际写入还会核对审批时的版本。决定用：

```bash
uv run lite-pi approve --session /absolute/path/to/session.jsonl --id APPROVAL_ID --decision approve
```

`reject` 和 `revoke` 同样只写回执。任务调用上限、输出预留和经过时间属于这个会话，续开、恢复、摘要及每次 HTTP 重试共用，不替代费用账本。无用量回执的尝试保留未知调用数与输出预留。持久化文件范围在重新打开会话后继续生效；`effect_unknown` 会持续阻止后续模型调用，当前没有单独的人工确认命令。

`code --delegate` 在 read/edit/write/bash 之外增加只读调查。一次调用最多两项，父任务须仍拥有 read 权限，调查路径取父授权范围的子集。每项最多四个普通文件，每个不超过 50 KiB；子会话只有 read，不加载项目指令、Skill、MCP 和扩展，读取固定在调查文件内，模型调用计入当前会话的任务预算。结果中的摘录由宿主在子会话返回后回读；子任务的完成声明不能当作文件已经改好。子报告中无法核对的引用会明确标记。调查结束后由父任务核对目标文件并自己修改。调查线程可并行，模型请求仍遵守全局最多一条在途请求的限制。裸命令 `lite-pi delegate` 返回拒绝，退出码 2，不会启动子任务。

`code --knowledge` 在 read/edit/write/bash 之外增加 `search_docs`、`rebuild_index`、`memory_propose`、`memory_read`。`--max-retrievals` 限制这一次运行的检索次数，默认 4。语料清单是项目内 `.lite-pi/corpus.json`，索引和记忆放在 `.lite-pi/knowledge.sqlite`。模型只能提出 pending 记忆。确认、撤回和冲突取舍用：

```bash
uv run lite-pi memory-confirm --session /absolute/path/to/session.jsonl --id MEMORY_ID
uv run lite-pi memory-revoke --session /absolute/path/to/session.jsonl --id MEMORY_ID
uv run lite-pi memory-resolve --session /absolute/path/to/session.jsonl --id MEMORY_ID
```

确认可以带 `--supersedes` 和正数 `--ttl`。有效记忆在下一次带 `--knowledge` 的运行里进入系统前缀，并带上来源、对象和版本。未确认的对话不会成为约定。资料正文和记忆正文不增加工具。

`branch` 只追加需求并选择历史，下一次 `resume` 才调用模型。`resume --leaf VALID_RECORD_ID` 可以选择其他有效历史。分支不回滚项目文件；缺少工具结果的中间记录不能作为续接点。

```python
from pathlib import Path
from lite_pi.session import AgentSession

session = AgentSession(Path("/absolute/path/to/project"), tools=("read",))
result = session.submit("读取实现，解释检查失败的原因。", max_turns=6)
print(result["status"], result["text"], result["evidence"])
```

公开会话可以设置轮次、真实请求次数、模型输出和 Shell 时间上限，也可以设置 `approvals`、`max_task_calls`、`max_task_output_tokens`、`max_task_seconds`。`decide(approval_id, decision)` 只记录审批。`context_tokens` 是这一次运行的上下文预算（最多 128000 tokens，包含输出预留）；估算超过预算减输出预留时才压缩，默认不设置，因此既有短任务不会压缩。`steer(text, paths=...)` 在下一次模型请求前生效，`paths` 由宿主限制后续写入。`follow_up(text)` 排到当前任务停止之后，并解除前一条文件范围。`cancel()` 停止当前模型和命令，不回滚已经写完的文件。`interrupt()` 在真实流开始后截断连接。压缩需要持久 Journal，摘要请求计入 `max_model_calls` 并响应 `cancel()`；摘要失败或取消时保留原历史。完整的最近消息组及必要的摘要超过窗口时明确停止。

### 语义与混合检索

BM25 不加载语义依赖。启用语义模式前，在 lite-pi 工程目录安装可选运行时并下载固定的本地模型：

```bash
uv sync --extra embeddings --group dev
uv run --extra embeddings python scripts/prepare_embeddings.py
uv run --extra embeddings lite-pi index --root /path/to/project --mode semantic
```

模型为 MIT 许可的 `intfloat/multilingual-e5-small`，固定 revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`。默认放在本工程 `.artifacts/models/multilingual-e5-small/<revision>/`；下载脚本可传 `--proxy`，显式指定公开下载代理。宿主可在 `index`、`code`、`resume`、`chat` 传 `--embedding-dir` 使用已有模型目录；该参数不暴露给模型工具。

`search_docs` 的 `mode` 为 `bm25`（默认）、`semantic` 或 `hybrid`，`limit` 为 1～4。semantic 使用真实 E5 单位向量做精确检索；hybrid 对最多各 8 个两路候选按片段身份去重，用 RRF（k=60）融合，再以 deepseek-flash 作一次最多 8 项的相关性重排，最终最多采用 4 片段。重排只接受已有身份的完整排列。hybrid 要求持久会话及父任务预算，重排没有工具；错误、取消或缺少模型/索引均明确返回，调用方不会自动切换模式。

向量保存模型、revision、384 维、前缀、masked mean、L2 归一化、分块契约及资料代。语义分块不截断，超过 800 字符的长行或超过 512 token 的输入会要求调整资料。清单变化使当前索引失效；正文变化排除失效片段，重建成功后才发布新代。重建失败保留历史；查询在重排后再次核对来源和当前代。缓存同时绑定模式、查询、limit、算法、表示契约及资料快照。

使用 `uv run lite-pi retrieval-trace --root /path/to/project` 或 `AgentSession.knowledge()` 查看完整候选、融合、重排回执及引用位置。`request_count` 保留循环及摘要请求口径，`total_request_count` 包含同次运行的重排和子调用；累计统计以费用账本及任务预算为准，本地推理单独记录 CPU 和墙钟用时。

### 计划与澄清

新持久会话可加 `--planning`，提供 `plan`、`question` 两个宿主工具。复杂任务可建立 3～5 步计划；简单问答可直接继续。计划显示交付条件、依赖、模型声明及工具证据，证据通过只表示对应工具观察成功，交付效果仍需检查。

```bash
uv run lite-pi chat --root /absolute/path/to/project --planning
uv run lite-pi plan --session /absolute/path/to/session.jsonl
uv run lite-pi answer --session /absolute/path/to/session.jsonl --id QUESTION_ID --version 2 --text '1'
uv run lite-pi resume --session /absolute/path/to/session.jsonl --approvals
```

等待澄清时，终端输入选项序号或自由答复，Enter 提交并重新核对执行条件；Esc 取消问题，Ctrl+D 空草稿退出并保留待答状态。CLI 的等待回执包含 `question.id`、`question.options` 和 `question.expected_version`；将 ID 和 expected_version 分别传给 `answer --id` 与 `--version`。这里采用当前工作流版本，需求的 `requirement_version` 单独管理。CLI `answer` 只保存答复，后续 `resume` 明确继续。答复不批准文件或命令；审批仍针对当前候选。重复、过期和跨分支答复被拒绝。工具批次中断尚未恢复时，先 `resume` 补齐记录，再答复。

修改要求可使用 `plan --change`，JSON 包含 `expected_version`、`reason`、`steps`、`invalidated_steps`；每个步骤包含 `id`、`title`、`depends_on`、`deliverable`、`inputs`。修订保留历史效果及独立步骤，失效步骤需要新证据；不自动回滚或重放文件写入。`plan` 和 `question` 由宿主保留，扩展不能替换这两个工具。

## 文件与 Shell 边界

文件工具拒绝项目外路径、符号链接、硬链接和私密路径。模型工具不能修改 `.lite-pi/extensions/` 与 `.lite-pi/mcp.json`，防止模型生成的文件在下次会话中变成宿主可执行配置。编辑使用唯一、无重叠的精确匹配，保留未修改字节及权限；`write` 默认拒绝覆盖。长行通过 `offset` 与 `column` 继续读取。

Shell 使用过滤后的独立项目副本，排除任意深度的 `.env`、`.env.*`、`.git`、`.artifacts`、`.ssh`、`.aws`，名称匹配忽略大小写。项目根 `.lite-pi` 整体不进入命令副本，Shell 创建该目录后的改动也不能发布到宿主；Skill 资源通过 `skill` 工具读取。容器没有网络和宿主凭据，根文件系统只读，只开放项目副本与临时目录。命令结束后先确认容器已删除，再核对宿主未发生变化并同步效果；清理或同步失败会终止当前循环，保留诊断和已发布路径。

当前 Shell 支持最多 64 MiB、10000 个条目的小型 Python 项目，运行时为镜像内的 Python 和 Bash。普通文件创建、修改、删除及目录增删可以同步；特殊文件、硬链接和符号链接修改会被拒绝。现有符号链接仅作为输入保留，项目外目标在容器内不可访问。输出最多保留 16 MiB，模型可见前 8000 字节，超量时终止命令。Docker 不可用会明确受阻。

项目根的 `AGENTS.override.md` 优先于 `AGENTS.md`，采用文件的正文、路径和 sha256 作为系统前缀进入模型请求；被覆盖文件只记录路径和 sha256。它们不写入每轮 Journal，也不增加工具权限。Skill 放在 `.lite-pi/skills/<名称>/SKILL.md`。目录里的名称、描述和哈希进入请求；正文和技能目录内的资源只在 `skill` 工具返回后进入请求。同名 Skill、损坏的 frontmatter 和指向技能目录外的资源会失败。指令、Skill 与资源共用文件工具的访问边界：拒绝任意目录层级的符号链接、硬链接、特殊文件与私密路径，单文件上限 50 KiB，超限明确报错。SKILL.md 的 YAML frontmatter 由安全解析器处理，支持引号、多行描述与 CRLF；name 和 description 必须为非空字符串，重复键拒绝。

扩展放在 `.lite-pi/extensions/<名称>/extension.py`，提供 `register(api)`。`api.register_tool` 登记的工具和原生工具使用同一次校验、路径范围和授权检查。`api.on` 可以挂 `tool_call`、`tool_result` 和 `context`。调用前 hook 可以拒绝或改写参数，改写后会再校验。结果 hook 只能在显示文本和结构化文本一致时替换可见结果，原始结果保存在该次运行的 `tool-raw`。上下文 hook 追加一条带来源的系统说明，计入上下文预算，不写入 Journal。扩展文件通过 Workspace 一次读取、计算哈希并加载，拒绝祖先符号链接和硬链接，单文件最多 256 KiB。初始化失败会在调用模型前停止，不留下半套工具。`api.write` 按实际写入路径检查当前 steering 范围。扩展抛出异常或返回无效结果时记录副作用未知，停止后续工具执行，保留调用 ID 和原始结果。

MCP 配置是 `.lite-pi/mcp.json` 的 `servers` 数组，字段为 `name`、必填的 `command`、`args`、`tools`，以及可选的正数 `timeout_seconds`、`origins` 和 `read_only_tools`。没有该文件就不连接 MCP。配置最多 50 KiB，沿用 Workspace 的链接与路径检查。首版只使用 stdio，SDK 为 `mcp==1.30.0`，协议为 `2025-11-25`。配置里的 `tools` 是直接披露的远端工具，名称变成 `mcp__<服务>__<工具>`；服务名禁止 `__`，完整名称最多 64 字符。声明 resources 能力的服务才增加 `list_resources` 和 `read_resource` 工具。工具和资源清单支持分页，保留远端 schema 的约束含义，更新时同步描述和 schema。资源内容作为工具结果进入后续请求，不增加 bash 或其他工具。

每次 MCP 结果都保留独立原始 JSON，包括远端错误、结构化结果和非文本内容。显示文本超过 8000 字符时截断，完整输出使用独立文件名，后续调用不会覆盖。`tools/call` 超时、取消或断线时保留副作用未知状态，停止后续执行，不自动重发。取消会中断本地等待，已发生的远端写入不回滚；随后由 SDK 关闭连接、回收它实际启动的子进程，不读取服务写出的 PID 执行信号操作。`run_end` 与最终回执在收尾之后发布。握手失败或启动中取消会清理已启动的所有服务并抛出带服务名的错误。

扩展与配置中的 MCP 程序属于用户信任的宿主代码，运行前需要自行审阅。扩展直接执行 Python，MCP 程序在宿主运行；它们拥有宿主进程的文件权限，不能靠 Workspace 或 steering 沙箱化任意代码。子进程环境只包含 PATH、PYTHONPATH、PYTHONDONTWRITEBYTECODE，以及调用时已经存在的 HOME、PLAYWRIGHT_BROWSERS_PATH、TMPDIR 和 LANG，不传入 API key。HTTP、SSE、streamable-http、OAuth、sampling、prompts、roots 和 elicitation 未实现。

浏览器扩展使用锁定的 `playwright==1.64.0`。先安装浏览器：

```bash
uv run playwright install chromium
```

MCP 服务入口为 `python -m lite_pi.browser.mcp_server --origin http://127.0.0.1:<port> --evidence <目录>`。默认使用 Playwright Chromium；需要本机 Google Chrome 时显式追加 `--channel chrome`，不会自动切换浏览器。MCP 配置的 `origins` 应与启动参数一致，`tools` 为 `open`、`observe`、`act`、`query`，`read_only_tools` 明确列出 `open`、`observe`、`query`。只读属性由可信宿主配置声明，工具名称或站点配置不会自动授予该属性。

首版适配受控订货页面：对象使用唯一的 `data-node`，提交表单含 `action_id`，`/state` 的订单回执使用 `data-order-id`。测试业务服务在 [tests/fixtures/shop.py](tests/fixtures/shop.py)；接入其他页面须实现对应的业务状态核对，不能仅凭一段成功文案确定副作用。页面和服务须遵守 GET/HEAD 无业务写入的约定。浏览器请求被限定在配置的 origin；子 frame、弹窗请求、WebSocket、service worker、下载及范围外资源均受限。MCP 进程仍属于可信宿主代码，此约束不构成任意第三方程序的操作系统沙箱。

观察编号由驱动生成，关联页面内容、表单值与实际元素句柄。页面或对象改变后，旧观察会被拒绝，模型须重新观察。响应丢失时保留 action ID；同轮只开放对应查询，查询回执命中该 ID 后才解除限制。查询失败或缺少匹配回执时保留 `effect_unknown`。SQLite 保存原始未知结果和后续查询的关联；结束、取消或跨进程续开不会重放点击。跨进程未决动作当前保守阻塞，尚不自动恢复浏览器进行查询。

## Docker 空闲后退出的排查

`docker info` 提示 socket 不存在时，先检查 Docker Desktop 是否退出，以及 `~/Library/Containers/com.docker.docker/Data/log/host/com.docker.backend.log` 中退出前的错误。Shell 的执行和验收依赖引擎；纯文档、静态检查与确定性机制测试可以继续。

若日志显示空闲计时结束后 VM 停止，唤醒时出现 `The storage device attachment is invalid`，可在 Docker Desktop 的 Settings → Resources 中关闭 Resource Saver，再启动 Docker。此设置避免自动停机和唤醒，空闲时仍会占用虚拟机资源；它是该故障的环境规避措施。恢复后核对原镜像和实际容器任务，并观察超过原空闲阈值的运行状态。保留镜像、卷和配置备份，避免用恢复出厂设置处理已有开发环境。[Resource Saver 官方说明](https://docs.docker.com/desktop/use-desktop/resource-saver/)。

## 验证

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy
uv run --group compare python tests/contracts.py
uv run python tests/acceptance.py all
```

前三项为静态检查，`contracts.py` 运行免费的确定性机制测试；`acceptance.py` 按分组组合机制测试与收费的真实 DeepSeek 验收，分别记录测试层级。可按 `session`、`tools`、`workspace`、`journal`、`interaction`、`context`、`extensions`、`recovery`、`knowledge`、`browser`、`regressions` 单独运行；分组与判定见 [测试说明](tests/README.md)。必验条件未命中时保留 `not-covered`，汇总返回非零。

运行状态目录的 `runs/` 保存产品回执，`spend/` 保存调用与账本；仓库 `.artifacts/acceptance/` 与 `.artifacts/contracts/` 保存每次独立运行的项目与客观判定。这些运行材料及 `.env` 均被 Git 忽略。

知识库由宿主管理 `.lite-pi/knowledge.sqlite`。数据库及侧文件不支持符号链接或硬链接；连接期间不要用其他进程替换该目录。记忆工具的 `source_ref=user` 由宿主绑定本次实际用户消息，保存可核查身份、原文和哈希。CLI 的确认和冲突取舍会先处理到期；同一对象存在未解决冲突时，所有相关记录均暂停采用。资料清单的成员或版本变化后需要重新建立索引。
