.. _agent-runtime:

Agent runtime and trajectories
==============================

The ``api`` planner supports per-request context policies, explicit skill
directories, configured models and delegated agents. Callers supply their task
and observations through the existing prompt/input interface. The runtime does
not recognize benchmark-specific inputs or define scoring rules.

Quick start
-----------

The offline example exercises the real runtime with scripted model replies,
synthetic images/video and synthetic token/cache counters. It needs no API key,
GPU, robot or model service:

.. code-block:: bash

   pip install -e ".[test,runtime]" imageio imageio-ffmpeg
   python -m examples.runtime.offline_demo --output /tmp/runtime-demo-001
   python -m examples.runtime.offline_demo --output /tmp/runtime-summary-001 --compress

Run from the repository root. Use a new or empty output directory each time;
existing traces are never appended to as a new run. Open
``/tmp/runtime-demo-001/report/index.html`` in a browser. These counters demonstrate
accounting and are not provider measurements or benchmark results.

For a configured model, use ``--runtime-config examples/runtime/runtime.yaml``
with ``--planner api`` after replacing ``YOUR_MODEL`` and installing the chosen
robot environment. Add the usual task/robot arguments. The Dashboard uses the
same flag and file. Python users pass ``RuntimeConfig`` to ``EmbodiedAgent`` or
``build_planner``. Only configured delegates require the ``runtime`` installation
extra; context policies, skills and traces work in a single-agent installation.

Configuration
-------------

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
       instructions: Review only the evidence explicitly supplied in your task.
       skill_paths: [./skills/review]
       tools: [read_skill]

YAML and JSON accept the same fields. Paths in configuration files are relative
to that file; paths passed from Python follow the caller's working directory.
In YAML, quote the disabled trace mode as ``mode: "off"`` so it remains a string.
Invalid fields, conflicting settings, missing skill directories and duplicate
skill names fail before external resources are started.

.. list-table:: RuntimeConfig fields
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type / default
     - Meaning
   * - ``llm``
     - ``LLMConfig | None`` / ``None``
     - Same provider/model/retry configuration as the existing planner.
   * - ``context``
     - ``ContextPolicy | None`` / ``None``
     - Default keeps history; ``recent_turns`` retains complete exchanges.
   * - ``context_engine``
     - Sync or async callback / ``None``
     - Python-only replacement history callback.
   * - ``context_engine_factory``
     - Zero-argument callable / ``None``
     - Python-only factory creating one callback per agent invocation.
   * - ``skill_paths``
     - Sequence of directory paths / empty
     - Explicit on-demand skill catalog for this agent.
   * - ``skill_max_bytes``
     - Positive integer / ``262144``
     - Maximum bytes returned for one skill resource, without truncation.
   * - ``trace``
     - ``TraceConfig`` / full, video enabled
     - ``mode`` is ``full``, ``metadata`` or ``off``.
   * - ``subagents``
     - Mapping of names to ``SubAgentConfig`` / empty
     - Explicit delegates; no automatic directory discovery.

``SubAgentConfig`` requires non-empty ``instructions`` and supports
``description``, ``model``, ``llm``, ``tools``, eager ``skills``, ``skill_paths``,
``skill_max_bytes`` and the same context fields. A child's unspecified context
configuration inherits its parent's strategy; histories and factory instances
remain separate. Tools and skill directories are explicitly selected, not
inherited. Fixed instructions and tool schemas are outside the context callback.

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

An equivalent ``async def`` is accepted. The callback receives a private copy of
the working history once per logical model request. Transport retries reuse the
same processed request and do not rerun compression or tools. Return a list of
SDK messages; the processed history becomes the SDK's working history.

Processing order is original-input recording, user context engine, existing
image-window/image-budget policy, then the actual model request. Original
observations remain in full traces even when removed before the first request.
The initial-image retention option remains available through ``LLMConfig``.

Retain the latest user input text and all pending tool feedback. Tool calls and
their results must remain paired, including parallel calls in the same response.
Invalid output or callback failure ends the request with a context error; the
runtime does not silently restore discarded history. ``recent_turns`` preserves
the initial task and latest user-input group in addition to the requested number
of recent complete response/feedback groups, so ``keep_turns`` is not a strict
message-count limit. Images still follow the planner's existing image policy.

For mutable state, pass ``context_engine_factory=make_context_engine``. The
factory is called separately for each root or child invocation, including
repeated/concurrent calls to the same named delegate. Direct callbacks should be
stateless. Built-in ``context``, direct callback and factory are mutually exclusive;
configuration files cannot import arbitrary Python callbacks.

``examples/runtime/custom_context.py`` shows async compression. Its
``await summarize_history(ctx, old_messages, llm=None)`` helper uses the current
model by default, or an explicit ``LLMConfig``. It shares ``ctx.usage`` and
``ctx.usage_limits``, uses the retry wrapper, and has no context hook of its own.
Summary requests appear as ``purpose=context_compression``. If a summary spends
the last request allowance, the main request is rejected. Provider-reported
usage from failed calls remains counted. External services called independently
by a custom callback are not automatically included in runtime accounting.

Skill tools
-----------

Each ``skill_paths`` entry is a directory containing ``SKILL.md``. A YAML
frontmatter ``name`` and ``description`` populate the skill index; without them,
the directory name and first non-empty description line are used. Only that
metadata enters instructions. The model calls:

.. code-block:: python

   read_skill(name="operation", resource="SKILL.md")
   read_skill(name="operation", resource="resources/checklist.md")

Results contain ``name``, ``resource``, ``content``, ``source`` and ``sha256``.
Content is read afresh on each call. Unknown names, paths outside the directory,
unreadable files and oversized resources return structured ``error`` values.
Oversize errors contain ``size_bytes`` and ``max_bytes``. Absolute paths and
symlinks escaping the directory are rejected. Skills do not execute scripts,
grant new tool permissions or create robot action steps.

An empty catalog registers no ``read_skill`` tool. An existing tool with the same
name conflicts with a configured reader. Each child gets its own catalog; it
cannot read a parent's skill unless that directory was explicitly configured
for the child. Existing ``EmbodiedAgent.run(skills=[...])`` and
``SubAgentConfig.skills`` remain explicit full-text preloading, using file paths.

Models, retries and optional VLA
--------------------------------

Root ``runtime.llm`` uses the existing ``LLMConfig`` fields, including provider,
model, endpoint, environment/Python credentials, cache options, image policy and
``parallel_tool_calls``. Do not also set root ``llm`` or explicit
``model``/``base_url``. A child uses either ``llm`` or the existing provider-prefixed
``model`` shorthand. An unspecified child model inherits the parent's resolved
model, endpoint and retry policy; explicit shorthand uses provider environment
credentials/endpoints and the inherited retry policy. Explicit child ``llm``
uses its own complete configuration, including image retention.

The default is **three retries after the first attempt**, at most four provider
attempts for transient failures. Authentication/invalid-request failures do not
trigger transport retries. Cancellation/timeouts remain effective during
backoff. A stream that has opened is not replayed, and completed tool calls are
never replayed by transport retries. Model-output validation retries are separate.

For supported built-in robots, ``--no-vla`` (Python startup
``enable_vla=False``) excludes VLA components before initialization and removes
VLA tool schemas/dispatch and corresponding prompt instructions. The planner LLM
and robot environment remain independently required. Default behavior keeps VLA
enabled. Flash replay rejects this combination before startup. MCP users control
remote VLA loading through their server startup; local tool filtering does not
unload a remote model.

``McpServer(tools=("observe", "finish"), ...)`` optionally restricts both exposed
schemas and dispatch to an explicit allowlist. ``tools=None`` retains all server
tools. This changes access, not the external server's model lifecycle.

Trajectories, media and accounting
----------------------------------

With runtime configuration, trace capture defaults to ``full``. Legacy entry
points without runtime configuration retain their previous recording behavior.
``metadata`` records lifecycle/request metadata and reported usage without full
message bodies; ``off`` disables runtime traces. Non-full views explicitly show
that inputs/outputs were not captured. Full traces contain user, model, tool and
skill text, so choose the appropriate mode for the data in the run.

The run directory contains ``trace/manifest.json``, append-only
``trace/events.jsonl``, normalized request/response/message/tool snapshots and
deduplicated ``trace/content/`` media. Existing ``run.log``, sanitized
``llm_requests.jsonl``, ``llm_errors.jsonl``, ``states.json`` and transcripts remain
available. Snapshot inputs are the provider-adapter inputs, not raw HTTP bytes.

Events include run/agent/turn/context boundaries, logical requests and attempts,
retries, tool calls and artifacts. A turn is one context/model/tool cycle, not a
robot action step. Root, child and compression requests have separate identities
and parent relationships. Logging receives event summaries without reconfiguring
global handlers for each child.

Open the running Dashboard's ``/trajectory`` page for the live view, or export:

.. code-block:: bash

   python -m rpent.cli.trajectory RUN_DIR --output REPORT_DIR

``REPORT_DIR`` must be new or empty. Open ``REPORT_DIR/index.html`` locally and
move the entire report directory when sharing. Export is read-only with respect
to the source run and does not start a robot. Live and offline views share the
same Python projection and browser component, including per-turn input/output,
tool details, parent/child timeline, retry states, media and usage totals.

Input token counts already include cache reads/writes; output token counts
already include reported reasoning. Total tokens are input plus output.
Cache ratio is total cache-read tokens divided by total input tokens, not the
average of per-turn ratios. Usage is counted once per request attempt, including
children and summaries; parent cumulative snapshots are not added again.
Missing usage/cache reports and zero denominators are shown as unknown/N/A with
coverage information. A failed request is not assumed to have cost zero.

Runtime action clips and episode videos follow the integration's recording
support. ``capture_video`` controls trace-related video capture independently of
Dashboard visibility. Episode videos may appear only after toolkit cleanup.
Known frame ranges link actions to clips; missing mappings are not inferred from
wall-clock timestamps. MCP runs without media remain usable. Missing/failed media,
unfinished runs and recording failures are shown explicitly. Exporting a running
run preserves its partial status and last observed event sequence.

Input-conversion migration
--------------------------

``rpent.context`` moved to ``rpent.data_convert``. Replace ``ContextDocument``
with ``TextDocument``, ``ContextBundle`` with ``PlannerInput``, and
``assemble_context`` with ``convert_planner_input``. Import ``load_skill`` from
``rpent.runtime.skills``. The public ``initial_context`` argument and rendered
planner inputs are unchanged. Conversion performs no filesystem retrieval or
history compression; those responsibilities belong to skill loading and the
runtime context engine respectively.
