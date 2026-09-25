.. _unified-agent:

Unified Agent Interface and Training Data
=========================================

Design principles
-----------------

RPent owns the VLM/LLM request loop, tool scheduling, context, skills, retries,
and traces. A benchmark owns environment reset, robot execution, observations,
and native scoring. An episode follows ``task -> model reply/tool call ->
environment action and observation -> next model request`` until the environment
ends, the agent calls ``finish``, or a budget is reached. The agent's reported
``finish`` status is not a benchmark success signal.

Each model request records the actual system instructions, messages, tool
schemas, and model settings sent at that point. The context engine retains
history by default. A callback can replace the history before every request;
it cannot change the fixed instructions or tool permissions. A logical model
request may have several network attempts. Training records keep those attempts
under one call rather than duplicating the training target.

User interface
--------------

The four main ``EmbodiedAgent`` inputs are:

* ``system_prompt`` supplies task-specific robot rules, coordinate frames, and
  action limits.
* ``skills`` preloads selected Markdown files for this task. ``skill_paths``
  exposes selected directories containing ``SKILL.md`` through ``read_skill``
  on demand. These paths extend ``RuntimeConfig.skill_paths``.
* ``local_tools`` or ``mcp_servers`` expose action and observation tools. Use
  ``LocalToolSpec`` when the benchmark and agent share a process; use
  ``McpServer`` for an existing MCP service. ``ToolResult`` can contain state
  text and camera images.
* ``RuntimeConfig(context_engine=...)`` or ``context=ContextPolicy(...)``
  controls history before each model request. Python callbacks receive and
  return PydanticAI ``ModelMessage`` lists. The SDK validates the current input
  and paired tool calls/results.

This example executes robot actions on the benchmark's environment thread and
does not require an MCP server:

.. code-block:: python

   from pathlib import Path

   from rpent.embodied_agent import EmbodiedAgent, LocalToolSpec
   from rpent.llm import LLMConfig
   from rpent.runtime import RuntimeConfig
   from rpent.tools.toolkit import ToolResult

   agent = EmbodiedAgent(
       mcp_servers=[],
       local_tools=[LocalToolSpec(
           name="move_eef",
           description="Move the end effector and return fresh state and camera images.",
           input_schema={
               "type": "object",
               "properties": {"pose": {"type": "array", "items": {"type": "number"}}},
               "required": ["pose"],
           },
       )],
       output_dir=Path("runs/task-001"),
       llm=LLMConfig(provider="openai", model="YOUR_MODEL"),
       runtime=RuntimeConfig(),  # full trace and append history by default
   )
   with agent.start_episode(
       "Place the red block in the bowl.",
       system_prompt="Use world-frame move_eef poses; inspect each new observation.",
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

A synchronous, thread-safe tool can instead provide a
``LocalToolSpec(handler=...)`` callback taking an argument dict and returning
``ToolResult``, then use ``agent.run(...)``. For remote tools, pass
``McpServer(name="robot", url="http://127.0.0.1:8000/mcp")``; stdio is also
supported. Local and MCP tools may be mixed if their public names do not
collide. Give every episode an isolated environment and output directory.
GPU simulators generally need process or cluster-worker isolation supplied by
the benchmark. ``start_episode.deferred_tools`` must include every local tool
without a handler.

A minimal custom context engine is:

.. code-block:: python

   def my_context(ctx, messages):
       return messages  # append history; or trim/summarize complete tool groups

   runtime = RuntimeConfig(context_engine=my_context)

See :doc:`agent_runtime` for callback constraints, async summarization, model
configuration, and trace modes.

Standard training data
----------------------

The JSON Schema at ``rpent/schemas/training_episode.v1.json`` defines a versioned
training episode. Each new episode directory contains ``episode.json`` and
SHA-256-deduplicated image files under ``assets/``:

* ``task`` identifies the benchmark and version, task ID, instruction, seed,
  and optional metadata.
* ``calls`` lists logical LLM calls in order. Each call includes the actual
  request system prompt, messages, image references, tool schemas and settings;
  the assistant response, tool calls and resulting tool feedback; status,
  usage, and every network attempt. ``cache_point`` content preserves prompt
  cache boundaries. ``purpose`` distinguishes root, child,
  and context-compression calls.
* ``outcome`` contains the benchmark's native success, score, and status.
  Failed episodes can also be retained; training jobs choose their own filter.
* ``provenance`` records the source run, capture and export times, source trace
  manifest checksum, and versions known at collection time. Do not infer an
  unknown collection version from the machine running the exporter.

Images may be supplied as bytes, base64, a file path, or an ``asset_ref``
relative to ``source_dir``. Conversion copies and verifies them. For a remote
image URL, the benchmark must first supply the actual image file. An episode
is published as a new directory only after validation; an existing directory
is never overwritten.

When a benchmark already owns every LLM input and output, convert them directly:

.. code-block:: python

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
                   "versions": {"agent_commit": "<collection commit>",
                                "simulator": "<simulator version>",
                                "scorer": "<scorer version>"}},
   )
   record = load_training_episode(path)  # validate schema and image checksums

``response`` may be a string for a plain assistant text response. A failed
model request uses ``status="error"`` and ``response=None``; its retry attempts
remain under the same call. Supply the final environment observation in that
call's ``tool_results`` if there is no later model request to carry it.

After a benchmark scores an RPent-run episode, convert its full trace directly:

.. code-block:: python

   from rpent.training_trace import convert_rpent_trace

   path = convert_rpent_trace(
       "runs/task-001",
       {"benchmark": "my_benchmark", "task_id": "pick-01",
        "instruction": "Pick the block", "seed": 0},
       outcome={"success": native_score.success, "score": native_score.score},
       output_dir="training/pick-01-seed-0",
       versions={"agent_commit": "<collection commit>"},
   )

Configure the agent with ``runtime=RuntimeConfig()`` or an equivalent full
trace. ``metadata`` and ``off`` traces cannot restore model inputs and outputs.
Conversion preserves attempts per logical request and copies the images
actually sent to the model. The request snapshot is taken at the provider
adapter boundary, not from raw HTTP bytes. A network failure with no response
has no assistant output suitable as a training target.

Dataset layout and versions
---------------------------

Standalone ``convert_episode`` calls produce validated episodes without a
dataset index. Use ``TrainingDataset`` for a collection. Its layout is:

.. code-block:: text

   <output-root>/<dataset-id>/<dataset-version>/
     manifest.json
     episodes/<split>/<benchmark>/<task-id>/<episode-id>/
       episode.json
       assets/<sha256>.<ext>

``split`` is ``train``, ``validation``, or ``test``. Task identifiers are
escaped in paths; the original values live in the JSON records. Episodes of
one benchmark/task pair must stay in the same split within a dataset version,
so runs of the same task cannot leak across splits.

``rpent/schemas/training_dataset.v1.json`` defines the manifest. It records
dataset identity and version, manifest and episode schema versions, timestamps,
exporter name/version/commit, dataset metadata, status, and an index of each
episode's path, split, task, score, and JSON SHA-256. Each episode contains its
own image checksums. ``verify()`` checks the index, episodes, and images.

The versions serve different purposes: ``schema_version`` changes for
incompatible format changes; ``dataset_version`` identifies a release;
``benchmark_version`` identifies task definitions; ``provenance.versions``
holds the collection agent commit and simulator, scorer, prompt, and skill
versions. Each request also retains the actual model identifier.
``producer.git_commit`` identifies the **exporter**, not the agent that
collected the run. Unknown collection versions cannot be recovered later.

.. code-block:: python

   from rpent.training_dataset import TrainingDataset

   dataset = TrainingDataset.create(
       "/data/training",
       dataset_id="robot-actions",
       dataset_version="2026-09.v1",
       producer={"name": "my-exporter", "version": "1.0",
                 "git_commit": "<exporter commit>"},
       metadata={"license": "internal", "description": "arm manipulation"},
   )
   path = dataset.add_rpent_trace(
       "runs/task-001",  # or add_episode(task, calls, episode_id=...)
       {"benchmark": "my_benchmark", "benchmark_version": "2026-09",
        "task_id": "pick-01", "instruction": "Pick the block", "seed": 0},
       split="train",
       outcome={"success": native_score.success, "score": native_score.score},
       versions={"agent_commit": "<collection commit>",
                 "simulator": "<simulator version>",
                 "scorer": "<scorer version>",
                 "prompt": "<prompt version>", "skills": "<skill version>"},
   )
   dataset.verify()
   dataset.seal()

The version directory must be new. On a filesystem that supports ``flock``,
a file lock serializes writers to a draft dataset, and the manifest is updated
atomically. ``seal()`` verifies and
freezes the version; use a new ``dataset_version`` to change samples or splits.
If export stops partway through, indexed episodes remain in a draft dataset;
inspect it and continue with ``TrainingDataset.open(path)``.

RoboDojo example
----------------

``examples/robodojo/eef_episode_policy.py`` declares RoboDojo's ``move_eef``
as a ``LocalToolSpec``. Isaac's owning thread executes movement and returns
robot state and three RGB views; RPent maintains one agent conversation and a
full trace. See ``examples/robodojo/README.md`` for installation, cache
preflight, and GPU-worker steps. After native scoring:

.. code-block:: bash

   python examples/robodojo/export_training.py \
     --experiment /path/to/scored-experiment \
     --output-root /path/to/training-data \
     --dataset-id robodojo-eef --dataset-version 2026-09.v1 \
     --split train --seed 0 --benchmark-version 2026-09 \
     --versions-file /path/to/versions.json \
     --exporter-commit abc123 --seal

The script joins native success/score from ``results/once-status.json`` with
the matching RPent trace and builds the dataset manifest. ``versions.json``
is a JSON object with collection-time keys such as ``agent_commit``,
``simulator``, ``scorer``, ``prompt``, and ``skills``; omit unknown values.
Replace ``abc123`` with the exporter's actual commit;
``--exporter-commit`` records it separately.
``--successful-only`` exports only native successes.
Older runs without full trace or the recorded task instruction cannot be
recovered from request-usage logs alone.

Errors, retries, and API records
--------------------------------

``RetryPolicy(max_retries=3)`` allows three network retries after the first
request. HTTP 408, 409, 425, 429, 5xx, provider API errors, and connection or
timeout errors can be retried; permanent HTTP 400, 401, and 403 failures are
not. An opened stream is never replayed, nor is an already completed robot
action repeated because a later model request retries.
``llm_requests.jsonl`` records each attempt's status, request shape, and
provider-reported usage. ``llm_errors.jsonl`` additionally stores the
redacted error response body, status code, and request identifiers. Full
trace stores normalized model adapter requests and successful responses. It
does not promise raw HTTP headers, byte-for-byte responses, or token usage
that the provider did not report.
