.. _agent-runtime:

Agent runtime 与轨迹查看
========================

``api`` planner 支持逐轮 context 策略、显式 skill 目录、模型配置和子 agent。
调用方通过现有 prompt／输入接口提供任务与观测；runtime 不识别各 benchmark
的专用输入，也不定义任务评分规则。

快速运行
--------

离线样例使用脚本化模型回复、合成图像／视频和合成 token/cache 数值，运行真实
runtime 链路，无须 API key、GPU、机器人或模型服务：

.. code-block:: bash

   pip install -e ".[test,runtime]" imageio imageio-ffmpeg
   python -m examples.runtime.offline_demo --output /tmp/runtime-demo-001
   python -m examples.runtime.offline_demo --output /tmp/runtime-summary-001 --compress

请从仓库根目录执行，每次使用新目录或空目录；已有 trace 不会混入新 run。
浏览器直接打开 ``/tmp/runtime-demo-001/report/index.html``。样例数值用于验证
统计与展示，不代表提供商实测数据或任务效果。

使用真实模型时，先修改 ``examples/runtime/runtime.yaml`` 中的 ``YOUR_MODEL``，
安装所选机器人的环境，再为 ``--planner api`` 添加
``--runtime-config examples/runtime/runtime.yaml`` 及通常的机器人／任务参数。
Dashboard 使用相同参数和配置文件。Python 中向 ``EmbodiedAgent`` 或
``build_planner`` 传入 ``RuntimeConfig``。只有配置子 agent 时才需要安装
``runtime`` extra；单 agent 的 context、skill 和 trace 功能不依赖 harness。

配置字段
--------

.. code-block:: yaml

   llm:
     provider: openai
     model: YOUR_MODEL
     retry: {max_retries: 3}
   context: {strategy: recent_turns, keep_turns: 8}
   skill_paths: [./skills/operation]
   skill_max_bytes: 262144
   trace: {mode: full, capture_video: true}
   subagents:
     reviewer:
       instructions: 只审查任务中明确提供的证据。
       skill_paths: [./skills/review]
       tools: [read_skill]

YAML 和 JSON 支持相同字段。配置文件中的路径相对于该文件解析；Python
传入的路径沿用调用方工作目录。YAML 中关闭 trace 请写 ``mode: "off"``，
用引号确保该值仍是字符串。未知字段、配置冲突、缺失 skill 目录和重名
skill 会在启动外部资源前报错。

.. list-table:: RuntimeConfig 字段
   :header-rows: 1
   :widths: 25 30 45

   * - 字段
     - 类型／默认值
     - 含义
   * - ``llm``
     - ``LLMConfig | None``／``None``
     - 复用既有 provider、model 和 retry 配置。
   * - ``context``
     - ``ContextPolicy | None``／``None``
     - 默认保留历史；``recent_turns`` 按完整交互裁剪。
   * - ``context_engine``
     - 同步或异步 callback／``None``
     - 仅 Python：返回下一轮历史。
   * - ``context_engine_factory``
     - 无参 callable／``None``
     - 仅 Python：每次 agent 调用创建独立 callback。
   * - ``skill_paths``
     - 目录路径序列／空
     - 当前 agent 显式声明的按需技能目录。
   * - ``skill_max_bytes``
     - 正整数／``262144``
     - 单个技能资源的读取上限；超限报错，不截断。
   * - ``trace``
     - ``TraceConfig``／full、视频开启
     - ``mode`` 可选 ``full``、``metadata``、``off``。
   * - ``subagents``
     - 名称到 ``SubAgentConfig`` 的映射／空
     - 显式子 agent，不自动扫描目录。

``SubAgentConfig`` 必须提供非空 ``instructions``，还支持 ``description``、
``model``、``llm``、``tools``、全文预加载的 ``skills``、``skill_paths``、
``skill_max_bytes`` 和相同 context 字段。子级未指定 context 配置时继承父级
策略，历史和 factory 实例仍各自独立。工具和技能目录需要显式声明，不自动
继承。固定 instructions 和工具 schema 不属于 context callback 的消息列表。

Context engine
--------------

.. code-block:: python

   from pydantic_ai import RunContext
   from pydantic_ai.messages import ModelMessage
   from rpent.runtime import RuntimeConfig
   from rpent.runtime.context_engine import recent_turns

   def my_context(ctx: RunContext, messages: list[ModelMessage]) -> list[ModelMessage]:
       return recent_turns(ctx, messages, keep_turns=4)

   runtime = RuntimeConfig(context_engine=my_context)

也支持等价的 ``async def``。每个逻辑模型请求前，callback 收到工作历史的
独立副本；网络重试复用已经处理好的请求，不重新压缩或执行工具。callback
返回 SDK 消息列表，处理后的结果成为 SDK 的工作历史。

处理顺序是记录原始输入、执行用户 context engine、应用现有图像窗口／预算
策略，最后发送实际模型请求。即使首轮就丢弃某张图片，full trace 仍保留
原始观测。``LLMConfig`` 中保留初始图像的选项继续生效。

必须保留最新用户输入的文本和待模型消费的工具反馈；工具调用与结果须配对，
包括同一回复中的并行调用。非法返回值或 callback 异常会结束本轮并记录
context 错误，不会静默恢复被丢弃的历史。``recent_turns`` 除保留指定数量
的近期完整回复／反馈组外，也保留初始任务和最新用户输入所在的完整组，因此
``keep_turns`` 不是严格的消息数量上限。图像仍遵循 planner 的现有策略。

有可变状态时请传入 ``context_engine_factory=make_context_engine``。每次
root／child 调用均新建实例，同名子 agent 被反复或并行调用时也不共享状态。
直接传入的 callback 应无共享可变状态。内置 ``context``、直接 callback 和
factory 三选一；配置文件不能自动 import 任意 Python callback。

``examples/runtime/custom_context.py`` 展示异步压缩。
``await summarize_history(ctx, old_messages, llm=None)`` 默认使用当前模型，
也可显式传入 ``LLMConfig``。它共享 ``ctx.usage`` 与 ``ctx.usage_limits``，
使用现有 retry wrapper，自身不挂 context hook。摘要请求在轨迹中标记为
``purpose=context_compression``。摘要用完最后一次请求额度后，主模型请求会
被拒绝；失败调用中提供商已报告的 usage 仍会计入。自定义 callback 独立
请求其他外部服务时，这些消耗不会自动进入 runtime 统计。

Skill 工具
----------

每个 ``skill_paths`` 目录必须包含 ``SKILL.md``。YAML frontmatter 的
``name``、``description`` 构成技能索引；没有这些字段时，使用目录名和文档
首个非空说明行。instructions 只加入这些元信息，模型按需调用：

.. code-block:: python

   read_skill(name="operation", resource="SKILL.md")
   read_skill(name="operation", resource="resources/checklist.md")

结果包含 ``name``、``resource``、``content``、``source`` 和 ``sha256``，
每次调用重新读取文件。未知名称、目录越界、文件不可读和内容超限返回结构化
``error``；超限错误含 ``size_bytes`` 和 ``max_bytes``。绝对路径和指向目录
外部的符号链接被拒绝。读取 skill 不执行脚本、不增加工具权限，也不产生
机器人动作 step。

空目录配置不注册 ``read_skill``；已有同名工具与新 reader 冲突时会报错。
各子 agent 使用自己的 catalog，只有明确配置了父级同一目录才可访问相应
技能。既有 ``EmbodiedAgent.run(skills=[...])`` 和 ``SubAgentConfig.skills``
仍接收文件路径，并保留全文预加载行为。

模型、重试与可选 VLA
--------------------

Root 的 ``runtime.llm`` 复用 ``LLMConfig``，包括 provider、model、endpoint、
环境变量／Python 凭据、缓存、图像策略和 ``parallel_tool_calls``。此时不要
再显式设置 root ``llm`` 或 ``model``／``base_url``。子 agent 的 ``llm``
与既有带 provider 前缀的 ``model`` 简写互斥。子级未指定模型时继承父级实际
模型、endpoint 和 retry；显式简写使用提供商环境凭据／endpoint，并继承
retry。显式子级 ``llm`` 使用自己的完整配置，包括图像保留策略。

默认是 **首次请求之后最多重试三次**，即最多四次 provider attempt，只重试
瞬态错误。认证失败或非法请求不触发网络重试；取消和总超时在 backoff 期间
仍生效。已经打开的流不重放，已完成工具也不会因网络重试重放。模型输出校验
重试与网络重试分别记录。

受支持的内置机器人可用 ``--no-vla``（Python 启动参数
``enable_vla=False``），在初始化前排除 VLA 组件，并移除相关工具 schema、
dispatch 和 prompt 要求。Planner LLM 和机器人环境仍各自需要配置。默认保持
VLA 开启；Flash replay 会在启动前拒绝关闭 VLA。MCP 用户通过远程 server
启动配置控制 VLA 加载；本地过滤工具不意味着卸载远程模型。

可选的 ``McpServer(tools=("observe", "finish"), ...)`` 用显式 allowlist 同时
限制 schema 暴露与 dispatch；``tools=None`` 保留 server 的全部工具。
这项配置改变访问权限，不控制外部 server 的模型生命周期。

轨迹、媒体与统计口径
--------------------

配置 runtime 后默认启用 ``full`` trace；没有 runtime 配置的旧入口保留原有
记录行为。``metadata`` 保存生命周期／请求元信息及已报告 usage，不保存消息
全文；``off`` 关闭 runtime trace。非 full 视图明确标记正文未采集。Full
trace 包含用户、模型、工具和 skill 文本，请按运行数据选择采集模式。

Run 目录包含 ``trace/manifest.json``、追加写入的 ``trace/events.jsonl``、
规范化的请求／响应／消息／工具快照，以及去重后的 ``trace/content/`` 媒体。
既有 ``run.log``、脱敏的 ``llm_requests.jsonl``、``llm_errors.jsonl``、
``states.json`` 和 transcript 保留。输入快照记录 provider adapter 接收的
输入，不宣称是原始 HTTP 字节。

事件涵盖 run／agent／turn／context 边界、逻辑请求及 attempt、重试、工具与
artifact。Turn 是一次 context／模型／工具循环，不等于机器人动作 step。
Root、child 和摘要请求有各自标识与父子关系；日志接收事件摘要，不为每个
child 重新配置全局 handler。

运行中的 Dashboard 可打开 ``/trajectory`` 实时查看，也可离线导出：

.. code-block:: bash

   python -m rpent.cli.trajectory RUN_DIR --output REPORT_DIR

``REPORT_DIR`` 必须是新目录或空目录。浏览器直接打开其中 ``index.html``，
分享时移动整个报告目录。导出不修改源 run，也不启动机器人。实时与离线视图
共用 Python 统计投影和浏览器组件，展示每轮实际输入输出、工具详情、父子
时间线、重试状态、媒体和 usage 汇总。

Input tokens 已包含缓存读写，output tokens 已包含提供商报告的 reasoning；
总量只等于 input 加 output。缓存命中率为总 cache-read tokens 除以总 input
tokens，不平均各轮百分比。每个 request attempt 只计一次，包含子 agent 和
摘要；不会再次叠加父级累计快照。缺失 usage／cache 和分母为零时显示未知／N/A
及覆盖范围，不把失败请求推断成零消耗。

动作片段和 episode 视频依赖具体 integration 的录制能力。
``capture_video`` 独立于 Dashboard 是否显示，控制 trace 所需视频采集。
Episode 视频可能到 toolkit 清理后才可用；已知帧范围支持动作定位，没有
映射时不根据墙上时钟猜测。无媒体的 MCP run 仍可查看执行轨迹。媒体缺失／
失败、run 未结束和记录失败均会明确展示；运行中导出的报告保留 partial
状态和导出时的最后事件序号。

输入转换接口迁移
----------------

``rpent.context`` 已迁移到 ``rpent.data_convert``。
``ContextDocument`` 改为 ``TextDocument``，``ContextBundle`` 改为
``PlannerInput``，``assemble_context`` 改为 ``convert_planner_input``。
``load_skill`` 从 ``rpent.runtime.skills`` 导入。公开参数 ``initial_context``
及渲染出的 planner 输入保持不变。转换函数不读取文件或压缩历史；这两项职责
分别由 skill 读取和 runtime context engine 承担。
