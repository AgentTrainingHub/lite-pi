# lite-pi 架构与边界

产品入口为 Python 3.12 的 `AgentSession` 与 `lite-pi` CLI，覆盖工具执行、持久会话、任务恢复与审批、检索和记忆、浏览器、有限委派、终端 UI、计划与澄清。框架对照和评估留在测试层。当前平台验证与运行限制见 [README](README.md)，测试方法见[测试说明](tests/README.md)。

## 交互前端

`tui.app` 只拥有 prompt_toolkit 的终端、输入缓冲、布局与按键。`tui.controller` 拥有一个守护工作线程，按次调用公开 Session；事件和完成回执带操作身份进入队列，由 UI 主线程消费。旧运行迟到事件被隔离。输入提交与运行结束在同一控制锁内交接，尚未进入请求的输入归回草稿，避免在下一次运行中静默重发。

`tui.display` 将快照投影为纯文本消息、工具块、审批摘要和欢迎图案，统一转义终端控制字符。`tui.catalog` 限定当前项目的会话发现与续开，拒绝链接和身份不符的历史。导航只读取 Journal 与 Session.inspect 的只读快照，不启动模型或 MCP，也不认领任务数据库。

UI 不直接执行模型参数、不计费、不清除审批或未知效果。Session 继续拥有所有资源收尾；最终 run_end 在原有持久事件中附带不含 events 的完成快照，使清理后抛出的异常仍可呈现已落盘回执。最终文件和观察事件保持一致。后台渲染失败会交回应用退出流程，先恢复终端，再最多等待工作者 15 秒收尾；超时返回明确诊断，持久状态仍需恢复核验；清理错误作为原异常的附注保留。

## 职责与依赖

```text
CLI / Python caller
        ↓
AgentSession → Journal → storage
            → TaskBook → TaskBudget / ProjectGate → SQLite
        ↓
Agent Loop → Provider → spend gate → DeepSeek
        ↓                      ↓
ToolRegistry                spend ledger → storage
   ├─ read / edit / write → Workspace → storage
   ├─ Shell → filtered ShellWorkspace → Docker → validated file publication
   ├─ Extension tools / hooks → Workspace API
   ├─ MCP host → SDK stdio → trusted subprocess
   ├─ search_docs / rebuild_index → CorpusIndex → knowledge sqlite
   └─ memory_propose / memory_read → MemoryLedger → knowledge sqlite
```

`AgentSession` 持有项目、工具授权、文件观察、会话历史和这一次运行的控制队列。它负责提交、持久续开、分支选择、`steer`、`follow_up` 与 `cancel`。`loop` 只推进模型/工具状态，记录实际请求次数并输出终态，不扫描整个项目判断任务成功。

`tools.contracts` 定义工具描述及目录接口；`ToolCatalog` 只提供安装、替换、撤销与取消信号，MCP 通过这个接口同步工具，不依赖 Registry 实现。扩展加载返回 `LoadedExtensions`，内部保存类型明确的工具、hook、来源与待撤销名称；`ExtensionHost` 只提供受宿主检查的写入与撤销操作。MCP 配置先完整校验为 `McpConfig`，再启动服务，重复名称与格式错误不会启动一半服务。开放的 JSON schema、消息和工具 payload 留在 JSON 边界。

`provider` 将真实响应转换为 `ModelResult`，在执行工具前验证协议结构。流式请求在 HTTP 读取循环里把新到达的正文增量交给会话；读完之后不再把完整文本切成增量。中断和取消都留下已经到达的文本，并关闭连接，不重试。`ToolRegistry` 用一份 `ToolDefinition` 绑定 schema 与执行函数，并在每次执行时核对当前授权和文件范围。参数校验采用 JSON Schema，嵌套非法值同样拒绝。事件和返回消息是独立快照，调用方修改不会污染后续历史。

一次运行发布 `run_start`、`turn_start`、`message_delta`、`message_end`、`tool_start`、`tool_end` 和 `run_end`，同时保留既有的 `model_response` 与 `tool_result`。`AgentSession` 负责解除扩展绑定并关闭 MCP，之后通过 `RunTrace` 生成唯一的 `run_end`。`RunTrace` 统一分配事件序号和时间，先原子保存 `events.jsonl`，再原子保存作为完成标记的 `result.json`，最后向调用方交付结束事件。两份文件不构成跨文件事务；读取方以 `result.json` 存在为完成依据。循环不写最终证据。运行异常保留诊断并向调用方抛出；清理失败在结束事件与最终回执中标记 `effect_unknown`。steering 默认一次取一条，插在当前助手回合之后、下一次模型请求之前。follow-up 只在没有待执行工具、也没有待处理 steering 时进入。带路径的 steering 从进入请求时起限制 edit/write，也拒绝把范围外的 Shell 改动发布回宿主；已发生的文件保持原样，取消不回滚。follow-up 进入后清除这个范围。取消第一次生效时丢弃尚未进入请求的队列，并阻止后续工具启动；正在执行的命令会被终止，容器随后删除。

授权了检索或记忆工具时，Session 打开项目级 `KnowledgeDB`（`.lite-pi/knowledge.sqlite`），把 `CorpusIndex` 和 `MemoryLedger` 做成普通工具交给 Registry。Session 不把自身或 Registry 传给这两个对象。关闭顺序是扩展解绑、MCP、索引所持的本地推理运行时、知识库、任务库，然后才写运行证据。知识库关闭失败把该次运行标为 `effect_unknown`。 打开数据库时以目录描述符检查 `.lite-pi` 祖先，并核对主库及 WAL、SHM、journal 侧文件；符号链接、硬链接和特殊文件均拒绝，首次创建、WAL 和 schema 初始化通过宿主路径锁串行化。SQLite 连接仍依赖路径，要求连接存续期间宿主管理该目录；任意外部进程替换目录或文件不在当前隔离保证内。schema v3 保留 v2 记忆约束，并为 generation 增加表示契约、向量表及查询 trace；v2 以部分唯一索引保证每个对象最多一条 active 记忆。用户来源由宿主绑定真实消息身份、原文和 SHA-256，写入 `memory_sources`；历史未登记的用户来源不能通过新确认；旧有效记录缺少来源时也会停止采用并给出诊断，可由宿主撤回。多方冲突均暂不生效，确认和取舍先处理到期，冲突候选保留期限。模型只能提出 pending 记忆；确认、撤回和冲突取舍由宿主方法完成，并在下一次运行的系统前缀里读取当时的 active 行。检索和记忆读取标记为 read_only、replay=unsafe。清单、索引库和扩展配置一样，不能由文件工具改写。

有 Journal 的会话由 Session 打开 `TaskBook`，由它管理意图、审批与所有权，并统一关闭所持资源。`TaskBudget` 使用同一个任务连接管理请求预留和结算；`ProjectGate` 单独拥有项目并发库连接；`task_schema` 管理建表与旧库迁移，`task_storage` 管理连接和事务。初始化失败会释放已经取得的资源；退出先释放并发槽，再生成状态快照，清理错误进入最终证据。

任务状态以 Journal header ID 关联，工具意图由 session ID、assistant 记录 ID、call ID 共同定位；同一 call ID 可出现在不同回合或分支；仅同一助手批次内重复 ID 被拒绝。显式选择未完成分支时先验证完整前缀，再恢复尾部工具批次。校验后的最终调用用 `PreparedCall` 表达，准入返回 `RunAdmission` 或 `HaltAdmission`。意图在执行前提交；工具结果先写 SQLite，再补入当前 JSONL 分支，最后按效果标为 settled 或 effect_unknown。效果未知的阻塞在续开后保留。旧库只有在每个 call ID 可唯一关联助手记录时自动迁移；有歧义则保留数据并停止。

可选的 `plan`、`question` 复用 TaskBook 的连接、事务和所有权。schema v4 添加按 Journal 祖先锚点选择的 workflow 快照及意图的要求世代，旧会话无计划时保持原行为，未知 schema 停止迁移。内部 `WorkflowState`、`Plan`、`PlanStep`、`Question` 使用领域类型，持久化及公开 JSON 回执在边界转换。每轮模型请求装配前重新核对计划输入，当前文件变化立即使相关依据失效。可用证据目录的 `input_paths` 列出回执实际核对过的文件，命令文本里的文件名不能充当读取证明。计划属于工作说明；宿主不另建 DAG 执行器。步骤声明与核验分开，工具成功只证明该次观察，文件版本、依赖及当前要求仍需满足。失效传播保留原证据；修订前意图不能重新激活新要求的步骤，计划元数据也不能自证执行成功。

澄清绑定要求版本、分支、实际问题及用户答复记录。用户答复和计划修订先保存 Journal 意图，再幂等采用到 SQLite；恢复先选择目标分支、采用未提交修订及答复，再检查输入版本。未配对工具批次必须先恢复才可追加答复。待答时停止模型请求，仅同一已产生批次中具有明确独立计划依据的原生只读调用可继续。问题和答复均不授予写入资格；修订使当前分支旧候选失效，已发生的文件效果与批准历史继续可查。TUI、CLI 和公开 Session 共用这些状态，查看状态不重新推理或扫描文件。

审批候选和待决定记录在一个事务内创建。批准后，先把状态改为 intent 再执行，崩溃恢复进入对账。原生文件工具把审批时的文件哈希传到实际写入处，目标版本改变就拒绝覆盖。`read_only` 表示是否只读，`replay` 只表达重试策略；原生 read/skill 和 MCP 资源读取明确只读，扩展默认有副作用，即使声明 safe 仍需审批并占修改槽。持久化范围在恢复工具和模型请求前重新装入 RunControl、Registry。内存会话不建任务库。原生文件写入前的效果预览失败作为工具错误回填，恢复对账无法读取目标时保留未知。文件已发布后的目录同步失败、工具原始证据写入失败都按效果未知处理。RunTrace 在调用用户事件回调前保留工具效果；Shell 的 cleanup_failed、runtime_error、workspace_sync_failed 同样保留为未核实效果；回调异常或后续取消不能清除未核实状态。内存会话也会拒绝在未知状态下继续提交。


`Journal` 管理追加记录、父节点与叶指针、分支投影和工具配对。压缩记录追加在链尾，投影使用最新摘要以及 `first_kept_id` 起的完整消息组。常规切点为用户边界；轮内切点为完整助手消息组的起点，另以 `turn_user_id` 绑定并保留该回合原始用户记录。工具结果不能独立作为切点，跨回合的用户身份会被拒绝；连续压缩不会因保留的用户记录而恢复之前已经省略的消息组。更早的原文仍留在文件里。消息投影同时提供对齐的源记录 ID，重复的用户文本不会影响边界选择；压缩边界须属于父节点的祖先链，完整原始链的工具配对也必须有效。会话先取得日志锁并验证目标历史，再加载扩展与启动 MCP。一次持久运行持有独立文件锁；项目并发库的首次初始化另外串行化，避免同时切换 WAL 模式的锁冲突。记录先 fsync，叶指针再原子替换。坏尾、格式不符、缺失父节点、重复 ID 和未配对调用直接产生诊断。会话 header 固定工具授权，重新打开允许缩小授权。分支切换只选择历史，不改变磁盘内容。没有 Journal 的内存会话不能压缩。

`CorpusIndex` 负责来源、资料代、缓存及查询落库，局部 `_rank_candidates` 汇总类型明确的排名结果；BM25、精确向量、RRF 与真实重排各有独立职责。`SearchHit` 保留来源版本和行号，`RankingTrace` 在序列化边界转成 JSON。`LocalEncoder` 只由显式语义模式惰性创建，索引负责释放，Session 先关闭索引再关闭共享数据库。ONNX、Tokenizer 和 NumPy 是可选依赖，模型仓代码不会执行。

`RunRequestBudget` 在原任务预算上增加本次运行的调用范围；循环、摘要、重排和子调查共享同一实例。父预算拒绝预留时不创建本地 ticket，未发送请求可释放，已发送的失败和未知用量仍计入原账本。一次重排直接使用 Provider、无工具、无重试，不建立第二个 Agent Loop。

## 文件与命令执行

`Workspace` 使用相对目录描述符和 `O_NOFOLLOW` 访问文件；私密名称在每层目录检查，硬链接与特殊文件拒绝。文件修改持有路径锁，写临时文件并 fsync 后原子替换，发布前比较当前内容与已读观察。权限位与未改字节保持原样。大小写别名共享观察和锁。文件工具拒绝修改 `.lite-pi/extensions` 和 `.lite-pi/mcp.json`；Shell 副本排除项目根 `.lite-pi`，并拒绝发布该目录，避免模型输出转化为下次运行的宿主代码。

Shell 的安全边界是固定摘要的 Docker 镜像及过滤项目副本。直接挂载 macOS 活目录再遮盖 `.env` 会留下大小写别名访问路径；过滤副本使这些对象从容器可见目录中消失，也避免宿主新增私密文件立即暴露。只在 Shell 调用时构建此工作区，范围限于 64 MiB、10000 个条目。

每个命令使用独立容器：非 root UID、无网络、无额外 capability、只读根文件系统、最多 64 个进程、256 MiB 内存和 1 CPU；临时目录为 16 MiB。命令进程不继承宿主环境。超时、取消或输出超量时终止容器，最终强制删除并记录状态。全部后代位于容器 PID namespace，包括 double-fork 与 setsid 后的进程。[Docker 运行参数](https://docs.docker.com/reference/cli/docker/container/run/)说明这些约束。

命令修改先留在独立副本。容器清理成功后，检查完整改动的路径和文件类型、宿主前后快照，再逐文件同步；保留已发布路径。同步并非跨文件事务，也不能排除最后一次核对之后任意外部进程再次修改文件。单独使用可丢弃项目、避免多个独立作者同时修改，是当前支持边界。同步中断或清理状态未知时停止循环，终态为 `effect_unknown`，不自动重试副作用。

## 请求与费用

`system_prompt` 提供带版本的宿主身份与当前工具说明。循环在工具源刷新后按当次公开 catalog 构建基础 system 消息，再附项目资源、记忆、扩展上下文和历史；当前工具以本次 catalog 为准。基础提示与资料一同参与窗口估算，压缩后重新装配，不写入或改写原始会话。摘要使用独立的 system 职责和无工具请求，原对话只作为待整理数据。提示中的权限说明由既有 Registry、Workspace 与 TaskBook 执行约束。

所有请求进入 `spend.gate`。持久会话先为每次 HTTP 尝试独立预留任务调用数和实际输出上限，再预留最坏费用，再持久化脱敏请求；发送前证据写入失败则释放预留。真实请求失败后无法取得用量时保留预扣并明确记录未知，实际模型身份与原始停止原因保留。

账本使用稳定侧锁与原子替换，不原地 truncate；空或损坏账本拒绝重建。预留记录 PID 与进程启动身份；发现所有者退出后保守计入未知费用，并释放并发槽。结算记录和金额一同原子写入账本，再导出 JSONL；导出失败时账本仍保留结算记录。账本金额是保守额度控制，官方余额为独立观察。

请求最多重试一次，每次重试重新申请任务预算；失败尝试没有用量时保留未知调用数和对应输出预留，后续成功不会覆盖它。发送前失败释放任务预留，发送后逐次结算；摘要复用同一入口。旧所有者不能新增、释放或结算任务预留。取消不重试。摘要与普通请求共享调用次数限制和取消信号；摘要完成前取消不会追加压缩记录，也不会启动后续请求。全局 2048 token 上限在唯一发送入口强制执行。余额判断扣除最近一次余额快照之后的已结算消费和在途预留。

## 当前支持范围

会话支持续开完整保存的工具批次，并在意图、效果和 SQLite 结果三个窗口恢复：safe 工具在两边策略都允许时重试一次，可查询文件按保存的哈希对账，无法核实则停在 `effect_unknown` 或 `awaiting_decision`。缺少结果的中间记录仍然拒绝作为普通续接点；只有尾部未完成的工具批次进入恢复。运行中支持增量输出、转向、后续任务和取消。在调用方给出的 `context_tokens` 低于估算输入加输出预留时生成真实摘要，并从项目根加载 `AGENTS.override.md` 或 `AGENTS.md`，以及 `.lite-pi/skills/<目录>/SKILL.md`。资源发现和读取复用 Workspace 的目录描述符与 O_NOFOLLOW 检查，同一次读取的字节用于内容解析和哈希。每个文本资源最多 50 KiB，YAML frontmatter 使用 PyYAML SafeLoader 并拒绝重复键。这些文本进入模型请求，不改变工具授权。扩展从项目 `.lite-pi/extensions/<名称>/extension.py` 登记工具和 hook，并从 `.lite-pi/mcp.json` 以 stdio 连接 MCP。扩展工具和远端工具都进入同一次 `ToolRegistry.execute`。每个模型回合重新披露当前工具。MCP 的工具清单与资源清单按游标分页读取；资源工具只在服务声明对应能力时披露。调用失败后的服务会被撤销，副作用未知时停止整个批次和后续请求，不自动重发 `tools/call`。每次远端结果的完整 JSON 与截断前文本使用独立证据文件。`delegate` 在进程内启动最多两个只读子会话。子会话跳过项目指令、Skill、MCP 与扩展；调查启动前检查父任务当前 read 授权和路径范围，只接受最多四个 50 KiB 以内的普通文件，固定 read_scope 不受后续 steering 清空影响，模型调用计入父任务的同一份 `TaskBudget`。委派调度、文件证据和具名回执分别位于 delegate、delegation_evidence 和 delegation_report。调查表只接收所需路径、取消事件、预算、事件回调及当前权限查询；子事件经队列在父线程统一写入 RunTrace。调查结束后恢复复用会话的历史与控制对象。宿主在子会话返回后重读文件并给出 sha256 摘录；完成声明不作为文件已改的证据。父会话在调查结束后再修改，父子不同时写同一项目。`delegate` 不在默认代码工具里，replay 为 unsafe。世代封印后迟到报告丢弃。会话收尾先收回调查线程，清理失败保持可见。取消和任务时间到期可中断正在握手或读取响应的 TLS socket；DNS/TCP 建连仍受连接超时约束。等待全局请求槽期间持续核对取消和时间预算；尚未发送则释放全局及任务预留，已发送但回调异常则按未知用量结算并保留原异常。原始部分响应、未知用量和预算预留按实际状态保留。调查线程最多两个，模型请求仍受全局在途上限约束。浏览器通过 MCP 接入，行为边界见后文。

扩展加载复用 Workspace；内容哈希与执行代码使用同一次读取的字节。工具 schema 在注册时验证并复制，调用结果的名称与 call ID 由宿主管理。`api.write` 按实际路径核对 steering 范围；上下文 hook 的说明计入预算。扩展异常和无效结果保留副作用未知状态。扩展与 MCP 服务是受信任的宿主代码，可直接使用 Python 或进程自己的文件接口，当前没有对任意扩展代码提供隔离沙箱。

MCP 每个服务持有独立线程、事件循环及 SDK ClientSession。请求在有界等待中响应取消；初始化的取消范围限制在连接上下文内部，使 SDK 能正常执行 EOF、终止和回收流程。宿主不接受服务提供的 PID 作为信号目标。公开服务报告含 `closed`，最终回执在清理之后刷新。测试通过独立观察文件核对进程是否退出，产品模块不读取这些文件。子进程环境只带 PATH、PYTHONPATH、PYTHONDONTWRITEBYTECODE，以及调用时已经存在的 HOME、PLAYWRIGHT_BROWSERS_PATH、TMPDIR 和 LANG，不带模型密钥。

浏览器扩展通过 MCP 连接，依赖保持为工具目录接口。业务服务拥有订单文件；`PageDriver` 通过 `AsyncExitStack` 拥有 browser/context/page，初始化失败和退出均执行已登记的收尾；MCP 服务在 driver 清理失败后仍关闭 Playwright。PID 留在内存，证据文件仅作输出。`DriverReceipt` 使用关键字字段、有限 Effect 类型和独立诊断。浏览器选择由 `--channel` 明确指定，默认 Chromium，无自动 fallback。

配置了 `origins` 的 MCP 服务接受宿主结果复核，但只读属性单独来自 `read_only_tools`，不根据工具名称推断。浏览器拦截每次 HTTP 请求，核对 origin 与主 frame；阻止子 frame、弹窗请求、WebSocket、service worker 和下载。页面对象使用 `data-node`，观察身份由宿主生成；捕获前后内容一致才发布，操作前复核 DOM、表单值与实际元素句柄，拒绝节点替换及旧观察。受控业务服务须遵守 GET/HEAD 无业务副作用约定，当前适配 `/state` 的 `data-order-id` 订单回执。

动作失败保留 action ID 和未知效果。MCP 将 `effect` 提升为宿主结果，通过 `effect_key`、`reconcile_tool` 标识允许的对账查询。循环在未决期间只执行对应查询；查询未解决就保持独立的 `effect_unknown` 终态，包括模型主动停止、预算耗尽或取消。查询结果必须匹配未决 action ID 才能产生 `resolves_effect`。TaskBook 在保存查询结果的同一事务中登记原意图与查询意图的关联，再同步会话日志；schema v3 的 `reconciliations` 保存关联，原始未知结果保持不变。崩溃后的 unsafe intent 不重放；跨进程未决动作保守阻塞，尚不自动重建浏览器对账。内存会话在同一 Session 对象上保留未知阻塞。

框架比较独立于产品执行链路。`tests/compare` 用 LangGraph 1.0.10 的 `model` 与 `tools` 节点调用现有 `provider.complete` 和 `ToolRegistry.execute`，依赖放在可选组 `compare`。图不调用 `AgentSession.submit` 或 `lite_pi.loop.run`。模型调用计入同一 `TaskBudget`；图里的工具不经过 `TaskBook.admit`。真实任务首轮后暂停，宿主检查反馈写入同一编译图、同一 `MemorySaver` checkpoint，再通过 `invoke(None)` 续接。每个工具单独提交状态，未知效果停止剩余批次，不能重放先前已经执行的工具。图执行抛错时，最终回执从最后一个已提交 checkpoint 保留已有工具结果，并禁止再次调用失败的执行实例。初始化、执行、资源清理与证据写入错误分别保留；清理和证据错误附在原异常上，不覆盖原异常。图只验证进程内 checkpoint，不提供产品完整的持久意图、审批和跨进程崩溃恢复。产品会话续接仍走 Journal。

上游 Pi 固定 commit 为 `7d75a0e3eb273dff753d49c066ea29fbe7b3b6f6`。基线比较脚本使用独立的 macOS sandbox 与本机费用代理；产品 Shell 使用 Docker。Python 实现的支持范围限于当前公开接口，不保证 Pi 日志格式、全部 Provider 或 TypeScript 插件兼容。来源与保留行为见[上游源码与行为依据](baselines/coding/upstream-map.md)。

索引代保存完整来源清单快照。查询同时核对清单成员、声明版本和文件内容哈希；清单删除、损坏或变化使旧代失效，失败重建保留历史代但不将其作为当前证据。内部来源、命中、过期记录与查询快照采用具名类型；查询复用同时包含查询文本、limit 和索引/文件快照。


## 展示与阅读

`tui.messages` 定义角色、工具状态及稳定消息块；`markup` 以 markdown-it-py 4.0.0 解析显示副本，`text` 在解析前后转义终端与双向控制字符。正文继承终端前景与背景，`theme` 集中控制标签、强调及状态颜色。代码块保持缩进，行内代码使用独立强调。

`TranscriptControl` 按可见行管理跟随与回看，使用块、源行和字符位置作为阅读锚点。稳定块的行缓存与宽度缓存独立，编辑缓冲始终由原应用持有；内容到达不抢回历史位置。PageUp/PageDown 保留一行重叠，Ctrl+End 恢复跟随。折叠隐藏当前位置时返回同块标题。

事件只投影成展示状态，执行仍归 AgentSession。Screen 统一提供阻塞判定与状态标签；Controller 只在显式展开或新工具结果需要展示时读取证据。读取器限定 RUNS 内同一 run，复用 Workspace 的描述符与无链接检查；原文缺失、无效及超限保持可见。每个文件内联读取最多64 KiB，已读原文完整呈现，超限保留完整证据位置；大回执显示有界预览，原文件保留。界面没有新增权限、费用结算逻辑或第二个工具执行器。

## 评估与诊断边界

评估工具位于 `tests/compare/`。`paired_manifest` 在任何模型请求前解码固定任务、顺序与共同预算，`paired_run` 驱动公开Session，`paired_evidence` 从原始父子回执、预算和实际请求重建观察；`paired` 只验证配对身份并计算任务级统计。未知用量不作零费用，缺失配对不发布完整集合结论。单Agent/团队的九配对也用于delegate可用性的单项消融，不重复计独立样本。

Judge 的 Reference 区分 human 与 ai 来源，默认要求实际人工确认；显式 mixed 模式接收具备授权记录的 AI 代评，并按来源分别统计。模式与原参照字段在请求前冻结，部分人工记录不能自动改用 AI。评分提示固定且无工具，参照标签不进入请求；候选标签不能自动成为人工标签。`judge_receipts` 核对实际请求、原始流、终态与费用计量；正式汇总和只读复算共用同一核对路径，拒绝缺失、重复运行及不一致的投影。RAG诊断使用公开开发题的实际检索，单独改变装配阶段输入，生成仍经过同一费用门禁。两者共用 `model_episode` 保存请求前输入、部分事件、预算和已发请求，发生异常时由调用方保留partial报告。诊断写入失败只附注到原异常。客观文件、权限、进程与引用检查继续独立于Judge。混合参照评分与完整人工校准分别记录，报告按参照来源列出结果与缺口。

## 工程边界

- `runtime.RuntimePaths` 在进程启动时读取显式配置；安装目录不保存可写状态。所有项目共用宿主账本、锁与运行证据，项目 Journal 保持项目归属。
- `run_settings` 统一公开运行限制；`cli_args` 负责语法与数值校验，`cli` 负责调用和呈现。`AgentSession` 负责准入与生命周期；`AgentLoop` 分别处理控制输入、模型轮次、工具批次和终态。
- `ToolControl`、`RunStatus`、`ControlInput` 定义宿主状态。原生结果通过带类型参数的 `tool_outcome` 构造控制字段，扩展内容与外部 JSON 在 Registry 边界验证；工具专属 reason 保持可扩展，通用循环按 `blocks_run` 和 `effect` 转移。
- `ApprovalStore` 负责审批记录和决定；TaskBook 负责意图、恢复及预算协调。`CorpusSources` 负责来源边界，`IndexBuilder` 发布索引世代，`QueryStore` 保存查询与核对新鲜度，`CorpusIndex` 组织排名。
- Session 注入只读子调查接口，DelegationTable 不反向依赖 Session。工具定义与执行策略分开；测试商店和崩溃注入仅位于 `tests/fixtures` 与测试入口。
- `StreamAssembler` 增量解析完整 SSE 行，异常流保留诊断；`StreamCapture` 负责受控写入、截断和状态。费用证据通过原子替换落盘，结算记录实际超出预留的差额。
- Session 收集每个资源的清理异常，保留首个运行错误及独立 `cleanup_errors`，全部清理之后发布最终证据。MCP 传输持有自己启动的进程组，正常关闭失败时强制回收；项目并发槽记录 PID 与启动身份。Linux 身份由 proc 启动时钟与 boot ID 组成，macOS 使用系统 ps；仍存活的 Linux 旧格式所有者保守占用槽，退出后再回收。

产品代码全量采用 strict mypy，Ruff 覆盖 B、SIM、RUF、C90；圈复杂度上限 25。CI 自动发现机制测试，环境缺失单列 blocked。真实模型验收、独立平台验证与机制测试分别记录。

宿主发给模型的固定系统说明、工具描述和边界提示统一为英文；用户内容与来源原文保留原语言，终端交互提示使用中文。模型最终答复遵循用户语言。协议标识符不做翻译。
