# RoboDojo with EmbodiedAgent

## RoboProbe EEF episode example

The new ``RoboDojo_EmbodiedAgent_EEF`` user policy keeps RoboProbe's
``move_eef`` parsing and CuRobo trajectory planner. RPent owns one continuous
LLM conversation per episode. The ``move_eef`` MCP call is handed to the
RoboDojo client process; its tool result is returned only after Isaac executes
the planned action and supplies a fresh robot state and three RGB views.
RoboDojo owns task termination and native scoring. This policy uses one model
response per accepted action, plus requests needed to repair rejected actions.

Prepare a fresh workspace with the existing ``roboprobe_once/prepare.py``,
then configure the EEF user demo. The runtime dependency directory must
contain ``mcp`` and ``pydantic_ai`` for Python 3.11. Keep the model key outside
Git; the configurator copies it into the experiment with mode ``0600``.

```bash
python roboprobe_once/prepare.py --experiment /path/to/new-experiment
python RPent/examples/robodojo/configure_eef_experiment.py \
  --experiment /path/to/new-experiment \
  --repo /path/to/RPent \
  --deps /path/to/python-dependencies \
  --key-file /private/path/to/tokenhub-key
python RPent/examples/robodojo/cache_gate_eef.py \
  --source-experiment /path/to/scored-reference-experiment \
  --experiment /path/to/new-experiment \
  --output /path/to/new-experiment/cache-gate-001
```

The cache gate makes multimodal RPent model requests using recorded RoboDojo
frames; it does not produce benchmark scores. It exits successfully only when
the provider reports a token-weighted cache hit rate above 60%. After the
gate and a bounded real task check, use the prepared ``manage.py`` to create
and submit separate GPU shards, for example ``plan --jobs 16``. The manager
keeps one native scored seed-0/layout-0 episode per task, and stores RPent
usage and retry logs under each task trace's ``rpent`` directory. This
54-task coverage run is distinct from RoboDojo's official multi-seed leaderboard.
After scoring, run ``manage.py status`` and ``report_eef.py --experiment
/path/to/new-experiment``. The report joins each native result to the same
run's RPent request log and reports per-task calls, retries, tokens, and cache
hit rate, plus the call-count median and P75.

## Discrete-action RoboDawn example

This adapter uses RPent's `EmbodiedAgent` for each model decision. Its MCP
`snapshot` tool returns the three RoboDojo camera views, robot state, and execution
feedback. A bounded RoboDawn demonstration enters the initial model context so
its images can be cached across decisions. `submit_action` accepts one to
four discrete commands. RoboDawn's controller plans and executes the commands
in Isaac Sim; RoboDojo owns reset and native success scoring. The agent's
`finish` report is not used as a score.

The RoboDawn controller and demonstration bank are third-party MIT-licensed
data. `install_policy.py` copies the pinned revision
`9247f366cd31f278e10f2fbe5fe8469b5f1b5b94` into an isolated benchmark
workspace and retains its license. No third-party assets or API credentials are
committed to RPent.

## Prepare the experiment

The following commands use the existing `roboprobe_once` workspace preparer
provided with this RoboDojo installation. Run from the parent directory of
`RPent` and set paths to match your installation:

```bash
python roboprobe_once/prepare.py --experiment /path/to/new-experiment
git clone https://github.com/Hugo-AGI/RoboDawn.git /path/to/RoboDawn
git -C /path/to/RoboDawn checkout 9247f366cd31f278e10f2fbe5fe8469b5f1b5b94
python RPent/examples/robodojo/install_policy.py \
  --source /path/to/RoboDawn --experiment /path/to/new-experiment
python RPent/examples/robodojo/configure_parallel.py \
  --experiment /path/to/new-experiment --repo /path/to/RPent \
  --key-file /private/path/to/tokenhub-key
```

`configure_parallel.py` adapts the prepared launcher's policy name, model
endpoint, and Python path. It copies the key into
`runtime/openai_api_key` with mode `0600`; neither the key nor the experiment
directory belongs in Git. Install RPent's runtime dependencies into
`runtime/python` for the Isaac image's Python 3.11 interpreter. The launcher
adds that directory and the RPent checkout to `PYTHONPATH`.
Warp, Torch extension, and CUDA caches are isolated per shard so 16 workers
cannot race while compiling the same kernel.

```bash
python3.11 -m pip install --target /path/to/new-experiment/runtime/python \
  'pydantic-ai-slim[openai]==2.48.0' 'mcp==1.30.0' \
  'openai==3.8.0' 'imageio>=2' 'prompt-toolkit>=3'
```

For an initial bounded check, pass `--decision-limit 4` to
`configure_parallel.py` and run distinct tasks in separate shards. A full
evaluation should use a fresh experiment with `--standard-tasks-only` and no
decision limit. This selects the 42 standard tasks that have pinned
demonstrations; the preparer also lists 12 random variants. The copied
`manage.py` supports `plan --jobs 16` and `submit`; each shard gets a separate
GPU worker. This setup scores seed 0 and layout 0, so it is not directly
comparable to multi-seed published results. The exact `sco` workspace,
storage mount, resource pool, and worker spec are installation-specific.
Use RoboDojo's native `_result.json`
`details[*].success` and `score` for evaluation, and keep the decision logs
under `runtime/vlm_logs` as diagnostic evidence.

The TokenHub `gpt-6-astra/azure_L/qwb` endpoint is configured through
`ROBODOJO_LLM_MODEL` and `ROBODOJO_LLM_BASE_URL`. The adapter uses its
Responses API because this endpoint rejects tool calls with
`reasoning_effort` on Chat Completions. RPent records each failed provider
attempt in the per-decision `llm_errors.jsonl` and exposes input, output, and
cached tokens in the decision response. HTTP 400 errors are logged and
returned; transient HTTP and connection errors follow RPent's retry policy.

The adapter sets a stable `prompt_cache_key` for the model route and enables
explicit Responses caching. RPent places breakpoints after the repeated task
and demonstration, and after multimodal `snapshot` feedback. It retains at most
two recent image observation groups in the planner history. The demonstration
and current cameras together remain subject to `max_images=24`.
Before submitting a full evaluation, replay recorded real multimodal decisions
and require a provider-reported token-weighted cache hit rate above 60%:

```bash
PYDANTIC_AI_NO_BANNER=1 python RPent/examples/robodojo/cache_gate.py \
  --source-experiment /path/to/prior-experiment \
  --run-prefix rp-once-prior-plan-id- \
  --output-dir /path/to/new-cache-gate \
  --key-file /private/path/to/tokenhub-key
```

The gate covers at least four decisions from each of three standard tasks and
writes `cache-gate.json` with the overall and per-task rates. It exits zero
only when every decision completes and the overall rate is strictly above
60%. This replay makes model requests but does not execute actions or provide
benchmark scores. Check account capacity for the full plan separately before
submitting workers.
Use the provider's reported cached input tokens to measure the weighted hit
rate for one plan:

```bash
python RPent/examples/robodojo/cache_report.py \
  --log-root /path/to/experiment/runtime/vlm_logs \
  --run-prefix rp-once-your-plan-id-
```

`cache_hit_rate` is `cached_input_tokens / input_tokens` for decisions with
usage data. Failed requests without provider usage are excluded; the report
shows their count as `decisions - decisions_with_usage`. During a live run,
`incomplete_decisions` counts JSON files that were still being written and are
excluded until the next report. Older adapter runs did
not forward cache write tokens, so `cache_write_tokens_reported=false` means
that figure cannot be recovered from those logs. The official OpenAI
[prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
describes explicit breakpoints and the response usage fields.
