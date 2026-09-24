EmbodiedAgent：接入外部 MCP Benchmark
=====================================

``EmbodiedAgent`` 将 RPent 现有的 planner 循环提供给 benchmark 使用，无需在
``robots/`` 下新增机器人包。benchmark 负责重置 episode、执行机器人动作、采集
观测以及判定任务成功。用户把动作和观测实现为 MCP 服务，提供 system prompt 和
可选的 skill 文件，然后对每个 episode 调用一次 ``run``。

.. code-block:: python

   from rpent.embodied_agent import EmbodiedAgent, McpServer
   from rpent.llm import LLMConfig

   agent = EmbodiedAgent(
       mcp_servers=[McpServer(name="robot", url="http://127.0.0.1:8000/mcp")],
       output_dir="runs/episode-001",
       planner="api",
       llm=LLMConfig(provider="openai", model="gpt-5.5"),
   )
   result = agent.run(
       "把红色方块放进碗里。",
       system_prompt=(
           "通过 MCP 工具控制机器人。move_eef 接收目标位姿并返回相机观测。"
           "每次操作后先检查结果，再决定下一步。"
       ),
       skills=["benchmark/SKILL.md"],
   )
   # 用 benchmark 自身的成功标志为这个 episode 计分。
   usage = result.stats["llm_usage"]
   print(usage["input_tokens"], usage["output_tokens"])
   print(usage["cache_read_tokens"], usage["cache_write_tokens"])

如果移动和拍照由独立的工具完成，也使用同一个入口；在 ``system_prompt`` 或
skill 中写明调用顺序即可。工具名、输入 schema 和描述直接从 MCP 服务发现。
planner 看到的名称为 ``<server_name>__<tool_name>``，MCP 返回的文本和图片
会传回模型。如需模型直接看见相机像素，请让工具返回 MCP 图片内容；纯文本
路径或 URL 仍按文本处理。任务说明、坐标系、动作限制以及观测规则由用户提供；
RPent 不预设 ``move_eef`` 或 ``snapshot`` 的参数格式。
单个 MCP 服务需要让模型直接看到 ``move_eef`` 等原始工具名时，可设置
``McpServer(expose_unprefixed=True)``。多个服务的工具名仍须唯一。

本地 stdio 服务使用 ``command``，并可提供 ``args``、``env``、``cwd``，
代替 ``url``。每个服务必须二选一；可接入多个服务，但名称必须不同。每次
stdio 服务继承父进程环境变量，``env`` 可以覆盖指定变量。每次 ``run`` 或
``start_episode`` 都会启动并关闭该服务。请为每个 episode 使用不同的
``output_dir``。

如果 Isaac 等仿真环境只存在于当前进程，可使用持续运行的 episode。MCP 服务
仍提供工具 schema；RPent 把指定工具调用交回持有环境的线程。应在环境执行动作
并取得新观测后，才调用 ``complete`` 返回工具结果。``ToolResult`` 可以通过
图片字节字段或 MCP 格式的 ``content_blocks`` 传回图像。任务是否结束及得分
均由 benchmark 决定。

.. code-block:: python

   from rpent.tools.toolkit import ToolResult

   with agent.start_episode(
       "把方块放到指定位置。",
       system_prompt="用 robot__move_eef 指定世界坐标系位姿。",
       deferred_tools=["robot__move_eef"],
       initial_context=["初始机器人状态：..."],
   ) as episode:
       while (call := episode.next_call(timeout=300)) is not None:
           observation = benchmark.execute_and_observe(call.arguments)
           episode.complete(
               call,
               ToolResult(call.name, {
                   "state": observation.state,
                   "_image_bytes": observation.camera_png,
                   "_finish": observation.episode_done,
               }),
           )
           if observation.episode_done:
               break
       result = episode.wait(timeout=300)
   native_score = benchmark.read_native_score()

``api`` planner 支持交回 benchmark 执行的工具。``start_episode`` 让多个
物理动作共用一次持续的模型对话，因此每个被接受的 ``move_eef`` 通常只需一次
模型响应。工具结果包含 ``_finish: true`` 时，对话无需额外模型调用即可结束。
benchmark 异常终止时，``EmbodiedEpisode.close`` 会释放等待中的调用。如果
机器人或相机连接由独立 MCP 进程持有，可使用 ``run``，由该服务直接执行工具。

``rpent.evaluation.embodied.evaluate_cases`` 接受独立的 benchmark 与 agent
工厂，并行运行不同样本，记录 benchmark 原生的 ``success`` 和 ``score``。
每个样本都必须有独立仿真实例；RoboDojo 使用独立 GPU worker 运行分片。

可选 planner 为 ``api``、``claude_code``、``codex``。``LLMConfig`` 用于
``api`` planner，支持 ``provider="openai"`` 和 ``provider="anthropic"``。
默认从 ``OPENAI_API_KEY`` 或 ``ANTHROPIC_API_KEY`` 读取凭据；也可显式传入
``api_key``、``base_url``。OpenAI 默认使用 Responses API；兼容 Chat
Completions 的端点可设 ``openai_format="chat"``。原有 ``model`` 和
``base_url`` 参数仍可使用，参见 :doc:`configure_planner`。

对支持显式 prompt 缓存的 Responses 端点，可在 ``LLMConfig`` 中设置稳定的
``prompt_cache_key`` 和 ``prompt_cache_mode="explicit"``。RPent 会在任务文字
及最近 40 条 Responses ``function_call_output`` 文本后设置缓存断点。设置
``image_history_groups=2`` 时仅保留最近
两组观测图像，较早图像改为文字占位。这些选项默认关闭，因为兼容端点不一定支持。
``preserve_initial_image_count=N`` 可让初始请求中的前 N 张示范图像在后续轮次保持
不变，而 ``image_history_groups`` 只限制之后的观测图像。机器人每次响应只
接受一个动作时，可设置 ``parallel_tool_calls=False``。
提供商报告的缓存命中率按已完成请求的
``cache_read_tokens / input_tokens`` 计算；输入 token 已包含缓存读取。

``result.stats["llm_usage"]`` 包含 ``input_tokens``、``output_tokens``、
``cache_read_tokens``、``cache_write_tokens``、``reasoning_output_tokens``、
``requests``、``cost_usd`` 和 ``total_tokens``。缓存输入已计入
``input_tokens``，推理输出已计入 ``output_tokens``，不要重复相加。若请求数
或费用未报告，则对应值为 ``None``；缓存数为 0 也可能只是提供商未报告。

独立调用模型可使用 ``LLMClient(LLMConfig(...))`` 的 ``generate`` 或
``generate_sync``；返回值包含 ``text`` 和本次 ``usage``，客户端的
``total_usage`` 则记录累计消耗。

LLM 请求遇到 HTTP 408、409、425、429、5xx，或没有 HTTP 状态码的
提供商 API、连接错误时会重试。
默认最多重试两次，并使用有上限的指数退避；可通过
``LLMConfig(retry=RetryPolicy(max_retries=...))`` 调整。HTTP 400、401、
403 等永久性错误会记录并立即返回。``api`` planner 每次失败的请求都会向
``<output_dir>/llm_errors.jsonl`` 写入状态码、尝试次数、重试决定和请求体量。
每次尝试（包括成功请求）还会写入同目录的 ``llm_requests.jsonl``。其中的
请求 ID 可关联重试；日志记录消息数、图片数、文字长度、图片字节数、缓存断点、
工具 schema 长度、耗时，以及提供商返回的 token 用量。这些尺寸有助于排查
失败请求，但无法代替失败请求未返回的精确 token 数。HTTP 错误还会记录
服务端错误码、类型、部分请求 ID 响应头及响应体 SHA-256 指纹。
``llm_errors.jsonl`` 对每次失败（包括 HTTP 400 和 500）记录服务端响应体；
可能回显 prompt、图像或凭据的字段会被遮蔽。没有响应体时，日志明确标记
``available: false`` 并保留异常消息。错误文件仅允许所有者读取。
``llm_requests.jsonl`` 只记录请求体量和用量，不保存响应体。独立调用可向
``LLMClient`` 传入 ``log_path``，
在该路径保存错误日志，并在同目录保存请求日志。独立调用的错误继续抛出；
``EmbodiedAgent.run`` 则通过 ``PlannerResult.error`` 返回。

每次运行都会重新读取并注入所列的 skill 文件，
不会自动加载工作目录中的文件。本地 ``finish`` 工具把 agent 的结论写入
``PlannerResult.finish_result``；评测得分应以 benchmark 的成功条件为准。

指定 memory 与初始观测
----------------------

通过 ``memory`` 传入已经筛选好的历史经验。每个 ``ContextDocument`` 保留
标题、正文，以及可选的来源标识：

.. code-block:: python

   from rpent.context import ContextDocument

   result = agent.run(
       "Place the red block in the bowl.",
       system_prompt="Use the registered robot tools and inspect each result.",
       skills=["benchmark/SKILL.md"],
       memory=[ContextDocument(
           title="Top grasp",
           text="Approach this object from above; recheck its current pose.",
           source="memory/grasp.md",
       )],
   )

Memory 以带标题的参考文本追加到任务后；提供 ``source`` 时也会显示来源。
它不会加入 ``system_prompt``。调用方负责筛选这些内容并确认其访问权限；
``source`` 仅用于记录来源，组装函数不会读取该位置。现有机器人流程的
memory 访问与合并规则保持不变。

``api`` planner 的 ``initial_context`` 支持文本和 PydanticAI
``BinaryContent`` 对象组成的序列。这些内容按传入顺序放在任务和所选 memory
之后，保留图像字节及其元数据。其他 planner 不支持此参数。Skill、memory
和观测统一通过 ``rpent.context.assemble_context`` 组装，CLI 和 Dashboard
也使用这个函数。直接使用结构化结果的方法见 :ref:`planner-context`。

配置子 agent
------------

安装 ``pip install -e ".[runtime]"`` 后，可以向 ``api`` agent 传入
``RuntimeConfig``。现有 ``run`` 参数仍用于配置主 agent：

.. code-block:: python

   from rpent.runtime import RuntimeConfig, SubAgentConfig

   agent = EmbodiedAgent(
       mcp_servers=[McpServer(name="robot", url="http://127.0.0.1:8000/mcp")],
       output_dir="runs/episode-001",
       llm=LLMConfig(provider="openai", model="gpt-5.5"),
       runtime=RuntimeConfig(subagents={
           "plan_reviewer": SubAgentConfig(
               description="检查拟执行的机器人计划。",
               instructions="指出给定计划中遗漏的必要前提。",
               tools=(),
           ),
       }),
   )

也可使用 ``runtime=RuntimeConfig.from_file("benchmark/runtime.yaml")``。
Python 配置中的 skill 路径按调用方工作目录解析；文件配置中的路径相对于配置
文件解析。子 agent 接收显式委派的任务文本，不自动接收父级会话或初始观测。
远程 MCP 动作工具由主 agent 使用；子 agent 只能选择实际存在于工具目录中的
现有 artifact／文本读取工具。工具限制、模型选择、共享 usage 和 CLI/Dashboard
用法见 :ref:`planner-runtime`。其他 planner 会在连接 MCP 之前拒绝 ``runtime``。

RoboDojo 示例
-------------

``examples/robodojo/eef_episode_policy.py`` 使用 RPent 的持续 episode
运行 RoboProbe 的笛卡尔动作与 CuRobo 规划逻辑。MCP ``move_eef`` 的结果包含
动作执行后的机器人状态与三路 RGB。``install_eef_episode.py`` 把这个用户
策略安装在新 RoboDojo 工作区中，``configure_eef_experiment.py`` 适配已有的
并行启动脚本。``cache_gate_eef.py`` 用录制的 RGB 通过相同的 RPent 请求链路
预检，要求提供商报告的输入 token 加权缓存命中率超过 60% 后再提交评测。
任务得分由 RoboDojo 原生评测给出。

此前的离散动作示例仍可使用：

``examples/robodojo`` 提供 XPolicyLab 适配层，使用 ``EmbodiedAgent`` 完成每轮
RoboDojo 决策。MCP 工具 ``snapshot`` 返回相机画面和机器人状态，
``submit_action`` 提交简短的离散动作；RoboDawn 的运动控制器负责执行，
RoboDojo 负责原生评分。示例固定 RoboDawn 控制器版本，将凭据保存在 Git 之外，
并提供并行 GPU 作业的启动适配。安装步骤与结果路径见示例目录的
``README.md``。
