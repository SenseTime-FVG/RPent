# 统一 Agent 接口与训练数据

原项目介绍与安装说明：[中文](README.zh-CN.md) · [English](README_old.md)。

## 设计原则

用户通过 `system_prompt`、`skills`、`tools/MCP` 和 `context_engine` 配置 Agent，各模块协作如下。

```text
user config: system_prompt / skills / tools/MCP / context_engine
                            |
                            v
+-----------+   task    +-------+  per-call context  +----------------+
| benchmark | --------> | Agent | <----------------> | context_engine |
| task/env  |           +---+---+                    +----------------+
+-----^-----+               | ^
      |           tool_call | | tool_response
      |                     v |
      +-- execute/observe +-----------+
                         | tools/MCP |
                         +-----------+

Agent -- LLM I/O --> Dataset -- standard format --> training set

task -> llm_response -> tool_call -> tool_response
     -> llm_response -> ... -> task finished
```

**Benchmark**：提供任务详情和工具执行所需的环境。

**Agent**：通过 `context_engine` 组织每次 LLM 调用的上下文。

**Dataset**：将 Agent 记录的每次 LLM 输入与输出整理为标准训练集格式。

## 用户接口

`EmbodiedAgent` 的四个主要入口是：

* `system_prompt`：每个任务的机器人规则、坐标系和动作限制。
* `skills`：本题全文预加载的 Markdown 文件；`skill_paths`：本题可通过
  `read_skill` 按需读取的 `SKILL.md` 目录。两者可与 agent 的
  `RuntimeConfig.skill_paths` 共用。
* `local_tools` 或 `mcp_servers`：向模型公开动作／观测工具。benchmark
  与 agent 在同一进程时，用 `LocalToolSpec`；已有 MCP 服务时用
  `McpServer`。工具结果可用 `ToolResult` 返回文本状态和图片。
* `RuntimeConfig(context_engine=...)` 或 `context=ContextPolicy(...)`：
  每次模型请求前组织历史。Python 回调接收 PydanticAI 的 `ModelMessage`
  列表并返回新列表；SDK 会检查当前输入与工具调用／反馈的配对。

下例在 benchmark 所在线程执行动作，因此不需要运行 MCP 服务：

```python
from pathlib import Path

from rpent.embodied_agent import EmbodiedAgent, LocalToolSpec
from rpent.llm import LLMConfig
from rpent.runtime import RuntimeConfig
from rpent.tools.toolkit import ToolResult

agent = EmbodiedAgent(
    mcp_servers=[],
    local_tools=[LocalToolSpec(
        name="move_eef",
        description="移动末端并返回执行后的机器人状态和相机图像。",
        input_schema={
            "type": "object",
            "properties": {"pose": {"type": "array", "items": {"type": "number"}}},
            "required": ["pose"],
        },
    )],
    output_dir=Path("runs/task-001"),
    llm=LLMConfig(provider="openai", model="YOUR_MODEL"),
    runtime=RuntimeConfig(),  # 默认 full trace 和 append 历史
)
with agent.start_episode(
    "把红色方块放进碗里。",
    system_prompt="使用世界坐标系的 move_eef；每次动作后检查新观测。",
    skill_paths=["skills/grasp"],
    deferred_tools=["move_eef"],
) as episode:
    while (call := episode.next_call(timeout=300)) is not None:
        observation = benchmark.execute_and_observe(call.arguments)
        episode.complete(call, ToolResult(call.name, {
            "state": observation.state,
            "_image_bytes": observation.camera_png,
            "_finish": observation.episode_done,
        }))
        if observation.episode_done:
            break
    result = episode.wait(timeout=300)
native_score = benchmark.read_native_score()
```

有同步、线程安全的工具处理函数时，可在 `LocalToolSpec(handler=...)` 提供
`dict -> ToolResult`，再调用 `agent.run(...)`。已有远程工具则传入
`McpServer(name="robot", url="http://127.0.0.1:8000/mcp")`；也支持 stdio。
一个 agent 可同时声明本地与 MCP 工具，名称必须互不冲突。每个 episode 使用
独立环境和输出目录；GPU 仿真通常要由 benchmark 使用进程或集群 worker 隔离。
`start_episode` 的 `deferred_tools` 必须覆盖所有没有 handler 的本地工具。

自定义 context engine 的最小形式是：

```python
def my_context(ctx, messages):
    return messages  # 默认 append；可按完整工具调用/反馈组裁剪或摘要

runtime = RuntimeConfig(context_engine=my_context)
```

完整的 callback 约束、异步摘要、模型与 trace 选项见 [Agent runtime](docs/source-zh/rst_source/usage/agent_runtime.rst)。

## 标准训练数据

训练 episode 使用 `rpent/schemas/training_episode.v1.json` 定义的 JSON
Schema。每个新目录包含 `episode.json` 与按 SHA-256 去重的 `assets/`：

* `task`：benchmark 名称与版本、task ID、instruction、seed 和扩展元数据。
* `calls`：按时间排列的逻辑 LLM 调用。每条包含实际请求的 system prompt、
  消息、图片引用、工具 schema、模型设置，以及 assistant 回复、工具调用、
  本轮工具反馈、请求状态、usage 和所有 attempt。消息中的 `cache_point`
  保留缓存边界；`purpose` 区分主 agent、
  子 agent 与 context 压缩请求。
* `outcome`：benchmark 原生 success、score 与状态。失败 episode 也可保存，
  由训练任务自行决定筛选规则。
* `provenance`：来源类型与 run ID、采集和导出时间、原始 trace manifest
  的 SHA-256，以及采集时各组件的版本。未知版本留空，不用当前导出环境猜测。

图片可以用字节、base64、文件路径或相对于 `source_dir` 的 `asset_ref`
传入；转换器会复制并校验内容。仅有远程 URL 的图片需要 benchmark 先提供
实际图像文件。文件采用新目录原子发布；同名目录不会覆盖。

当 benchmark 自己持有逐次 LLM 输入输出时，直接使用转换接口：

```python
from rpent.training_data import convert_episode, load_training_episode

path = convert_episode(
    {"benchmark": "my_benchmark", "task_id": "pick-01",
     "instruction": "Pick the block", "seed": 0},
    [{
        "call_id": "llm-1",
        "request": {
            "system_prompt": "Use move_eef.",
            "messages": [{"role": "user", "content": [
                "Pick the block",
                {"type": "image", "media_type": "image/png", "data": camera_png},
            ]}],
            "tools": [{"name": "move_eef", "description": "Move arm",
                       "input_schema": {"type": "object", "properties": {}}}],
        },
        "response": {"message": {"role": "assistant", "tool_calls": [
            {"id": "action-1", "name": "move_eef", "arguments": {}},
        ]}},
        "tool_results": [{"role": "tool", "name": "move_eef",
                         "tool_call_id": "action-1", "content": "arrived"}],
    }],
    outcome={"success": True, "score": 1.0},
    output_dir="training/pick-01-seed-0",
    provenance={"source_run_id": "benchmark-run-01",
                "versions": {"agent_commit": "<采集时的 commit>",
                             "simulator": "<仿真环境版本>",
                             "scorer": "<评分器版本>"}},
)
record = load_training_episode(path)  # 校验 schema 与全部图片哈希
```

`response` 可以是字符串，表示纯文本 assistant 回复。失败模型请求设置
`status="error"`、`response=None`；重试 attempt 留在同一 call 的
`attempts`。没有后续模型请求的最后一次环境反馈，应放在本次 call 的
`tool_results` 中，避免丢失终止观测。

RPent 自己运行的 episode 可在 benchmark 原生评分后直接转换 full trace：

```python
from rpent.training_trace import convert_rpent_trace

path = convert_rpent_trace(
    "runs/task-001",
    {"benchmark": "my_benchmark", "task_id": "pick-01",
     "instruction": "Pick the block", "seed": 0},
    outcome={"success": native_score.success, "score": native_score.score},
    output_dir="training/pick-01-seed-0",
    versions={"agent_commit": "<采集时的 commit>"},
)
```

必须在 agent 中传入 `runtime=RuntimeConfig()` 或等价的 full trace 配置。
`metadata`／`off` 无法恢复模型输入输出。转换保留每个逻辑请求的 attempt，
并复制 trace 中实际发给模型的图像。该快照位于 provider adapter 边界，
不是 HTTP 原始字节；网络失败且没有返回正文时不存在可训练的 assistant 输出。

## 数据集目录与版本

单独调用 `convert_episode` 会生成可验证的 episode，但不会建立数据集索引。
正式收集使用 `TrainingDataset`，目录固定为：

```text
<output-root>/<dataset-id>/<dataset-version>/
  manifest.json
  episodes/<split>/<benchmark>/<task-id>/<episode-id>/
    episode.json
    assets/<sha256>.<ext>
```

`split` 只能是 `train`、`validation` 或 `test`。路径中的任务标识会
转义，原始值以 `episode.json` 和 `manifest.json` 为准。同一数据集版本中，
相同 benchmark 与 task ID 的 episode 必须进入同一个 split，避免按 run
随机切分导致同题泄漏。不同版本的数据集互不覆盖。

`manifest.json` 的格式见 `rpent/schemas/training_dataset.v1.json`。
它保存数据集 ID 与版本、manifest/episode schema 版本、创建和更新时间、
导出程序的名称／版本／代码 commit、数据集元信息、状态及每条 episode 的
相对路径、split、任务和评分摘要、`episode.json` 的 SHA-256。图片哈希保存在
各 episode 内。`verify()` 同时核对索引、episode 内容和全部图片。

版本字段各有用途：`schema_version` 只在格式发生不兼容变化时升级；
`dataset_version` 标识一次确定的数据集发布；`benchmark_version`
标识题目定义；`provenance.versions` 记录采集时的 agent 代码 commit、
仿真器、评分器、prompt 和 skill 版本；每次请求还记录实际模型名。
`producer.git_commit` 是**导出程序**的 commit，不能替代采集时的 agent
commit。采集时版本若没有记录，后续转换无法可靠推断。

```python
from rpent.training_dataset import TrainingDataset

dataset = TrainingDataset.create(
    "/data/training",
    dataset_id="robot-actions",
    dataset_version="2026-09.v1",
    producer={"name": "my-exporter", "version": "1.0",
              "git_commit": "<导出程序 commit>"},
    metadata={"license": "internal", "description": "arm manipulation"},
)
path = dataset.add_rpent_trace(
    "runs/task-001",  # 也可用 add_episode(task, calls, episode_id=...)
    {"benchmark": "my_benchmark", "benchmark_version": "2026-09",
     "task_id": "pick-01", "instruction": "Pick the block", "seed": 0},
    split="train",
    outcome={"success": native_score.success, "score": native_score.score},
    versions={"agent_commit": "<采集时的 commit>",
              "simulator": "<仿真环境版本>",
              "scorer": "<评分器版本>",
              "prompt": "<prompt 版本>", "skills": "<skill 版本>"},
)
dataset.verify()
dataset.seal()
```

创建新版本时目录必须不存在。同一支持 `flock` 的文件系统上，多个 worker
可向同一 draft 版本追加 episode；写入采用文件锁并原子更新 manifest。
`seal()` 校验后封版，接口拒绝后续追加；
修改样本或切分应创建新的 `dataset_version`。导出中途失败时保留 draft
及已索引的样本，检查后可用 `TrainingDataset.open(path)` 继续追加。

## RoboDojo 接入示例

`examples/robodojo/eef_episode_policy.py` 将 RoboDojo 的 `move_eef`
定义为 `LocalToolSpec`，让 Isaac 所在线程执行运动并在反馈中提供状态与三路
RGB；RPent 持续运行一个 agent 对话并采集 full trace。安装、缓存预检与
GPU worker 启动步骤见仓库的 `examples/robodojo/README.md`。原生分数产生后：

```bash
python examples/robodojo/export_training.py \
  --experiment /path/to/scored-experiment \
  --output-root /path/to/training-data \
  --dataset-id robodojo-eef --dataset-version 2026-09.v1 \
  --split train --seed 0 --benchmark-version 2026-09 \
  --versions-file /path/to/versions.json \
  --exporter-commit abc123 --seal
```

该脚本将 `results/once-status.json` 的原生 success/score 与对应 RPent trace
关联，并创建上述数据集目录和索引。`versions.json` 是采集时版本的 JSON
对象，例如 `{"agent_commit":"...","simulator":"...","scorer":"...","prompt":"...","skills":"..."}`；
未知值可省略。将示例中的 `abc123`
替换为导出程序的实际 commit；`--exporter-commit` 单独记录它。
`--successful-only` 可只导出成功 episode。旧 RoboDojo 运行若没有
full trace 或记录任务 instruction，不能凭请求用量日志恢复训练样本。

## 错误、重试与 API 记录

`RetryPolicy(max_retries=3)` 表示首次请求后最多三次网络重试。
HTTP 408、409、425、429、5xx、提供商 API 错误和连接／超时错误可重试；
HTTP 400、401、403 等永久性错误不重试。已开始消费的流不重放，已经完成
的机器人动作也不会因模型请求重试而重复执行。
`llm_requests.jsonl` 逐 attempt 记录状态、请求体量和提供商报告的 usage；
`llm_errors.jsonl` 额外保存脱敏后的错误返回正文、状态码和请求标识。
Full trace 保存模型适配器的请求与成功响应快照。它不承诺保存原始 HTTP
headers、逐字节响应或提供商未返回的 token 用量。
