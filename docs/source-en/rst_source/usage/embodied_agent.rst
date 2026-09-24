EmbodiedAgent: external MCP benchmarks
======================================

``EmbodiedAgent`` exposes RPent's existing planner loop to a benchmark without
requiring a robot package under ``robots/``. The benchmark owns episode reset,
robot actions, observation capture, and success scoring. It exposes actions and
observations through an MCP server, supplies a system prompt and optional skill
files, then calls ``run`` once per episode.

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
       "Place the red block in the bowl.",
       system_prompt=(
           "You control the robot through MCP tools. The move_eef tool accepts "
           "a target pose and returns a camera observation. Check each result "
           "before choosing the next action."
       ),
       skills=["benchmark/SKILL.md"],
   )
   # Read the benchmark's own success flag to score this episode.
   usage = result.stats["llm_usage"]
   print(usage["input_tokens"], usage["output_tokens"])
   print(usage["cache_read_tokens"], usage["cache_write_tokens"])

The same entry point works when motion and camera capture are separate tools:
describe their intended sequence in ``system_prompt`` or a skill. Tool names,
input schemas, and descriptions come directly from MCP discovery. The planner
sees each as ``<server_name>__<tool_name>``. MCP text and image content are
passed back to the model. Return camera frames as MCP image content when the
model must see pixels; a text path or URL remains text. Include the task,
coordinate frame, action limits, and observation rules in the supplied context;
RPent does not assume a fixed ``move_eef`` or ``snapshot`` schema.
Set ``McpServer(expose_unprefixed=True)`` for a single server when the model
must see the tool's original name, such as ``move_eef``. Tool names must remain
unique across all connected servers.

For a local stdio server, pass ``command`` and optional ``args``, ``env``, and
``cwd`` instead of ``url``. Exactly one of ``url`` or ``command`` is required
per server. Multiple servers may be supplied, with unique names. A stdio
server inherits the parent process environment, with ``env`` values overriding
individual variables. It is launched and closed for every ``run`` or
``start_episode``. Use a distinct
``output_dir`` for each episode.

When the benchmark's simulator exists only in its current process, use a
persistent episode. The MCP server still supplies the tool schema; the named
tools are executed by the benchmark thread after RPent hands back each call.
Complete a motion only after the environment has stepped and captured a fresh
observation. ``ToolResult`` accepts image bytes through its image fields or
through MCP-style ``content_blocks``. The benchmark decides when the episode
ends and reads its own native score.

.. code-block:: python

   from rpent.tools.toolkit import ToolResult

   with agent.start_episode(
       "Place the block.",
       system_prompt="Use robot__move_eef to command world-frame poses.",
       deferred_tools=["robot__move_eef"],
       initial_context=["Initial state: ..."],
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

The ``api`` planner supports deferred tools. ``start_episode`` holds one model
conversation across all physical actions, so each accepted ``move_eef`` normally
uses one model response. A tool result with ``_finish: true`` ends that
conversation without another model call. ``EmbodiedEpisode.close`` releases a
pending call when a benchmark ends unexpectedly. For a separate MCP process
that owns the robot or camera connection, use ``run`` and implement tool
execution directly in that server.

``rpent.evaluation.embodied.evaluate_cases`` accepts independent benchmark and
agent factories, runs separate cases concurrently, and records the benchmark's
native ``success`` and ``score``. Each case needs its own simulator instance;
RoboDojo uses a separate GPU worker for each shard.

Supported planners are ``api``, ``claude_code``, and ``codex``. ``LLMConfig``
works with the ``api`` planner and supports ``provider="openai"`` or
``provider="anthropic"``. It reads ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``
by default; ``api_key`` and ``base_url`` may also be passed explicitly. OpenAI
uses the Responses API by default; set ``openai_format="chat"`` for a Chat
Completions-compatible endpoint. The previous ``model`` and ``base_url``
arguments remain available as described in :doc:`configure_planner`.

For a Responses endpoint that supports explicit prompt caching, set
``prompt_cache_key`` to a stable value and ``prompt_cache_mode="explicit"`` in
``LLMConfig``. RPent marks the task text and the latest 40 Responses
``function_call_output`` texts as cache breakpoints. Set
``image_history_groups=2`` to retain the last two
observation groups; older camera images become text placeholders. These
options are opt-in because compatible endpoints may not support them.
``preserve_initial_image_count=N`` keeps the first N demonstration images in the initial
request unchanged across turns while ``image_history_groups`` applies to later
observations. Set ``parallel_tool_calls=False`` when the robot expects one
action per model response.
Calculate the provider-reported cache hit rate as
``cache_read_tokens / input_tokens`` over completed requests. The input count
already includes cached tokens.

``result.stats["llm_usage"]`` contains ``input_tokens``, ``output_tokens``,
``cache_read_tokens``, ``cache_write_tokens``, ``reasoning_output_tokens``,
``requests``, ``cost_usd``, and ``total_tokens``. Cached input is included in
``input_tokens``; reasoning output is included in ``output_tokens``. ``None``
for requests or cost means it was not reported. A zero cache count may mean the
provider did not report cache usage.

For standalone model calls, ``LLMClient(LLMConfig(...))`` offers ``generate``
and ``generate_sync``. Each response has ``text`` and ``usage``; the client also
exposes cumulative ``total_usage``.

LLM requests retry HTTP 408, 409, 425, 429, and 5xx responses, plus provider
API or connection errors without an HTTP status. The default is two retries
with capped exponential backoff;
configure it with ``LLMConfig(retry=RetryPolicy(max_retries=...))``. HTTP 400,
401, 403, and other permanent failures are logged and returned immediately.
Each failed provider attempt writes a sanitized record to
``<output_dir>/llm_errors.jsonl`` for the ``api`` planner. The record includes
the status, attempt number, retry decision, and request shape. Every attempt,
including successful ones, is also written to ``llm_requests.jsonl`` in the
same directory. Its request ID links retries; it records message and image
counts, text length, image bytes, cache points, tool schema length, latency,
and provider token usage when available. These sizes help investigate failed
requests; they are not token counts for requests without provider usage.
For HTTP failures, logs also include the server error code/type, selected
request ID headers, and a SHA-256 fingerprint of the response body.
``llm_errors.jsonl`` records the response body for every failed attempt,
including HTTP 400 and 500. Fields that can contain echoed prompts, images,
or credentials are redacted. When the provider returned no body, the record
says ``available: false`` and includes the exception message. Error files are
written with owner-only permissions. ``llm_requests.jsonl`` stores request
shape and usage without response bodies. For direct calls, pass ``log_path``
to ``LLMClient`` to save the error records at that path and request records
beside it. Errors still propagate from direct calls;
``EmbodiedAgent.run`` returns them in ``PlannerResult.error``.

Skill files are read and inserted into the prompt on every run; they are not
automatically discovered from the working directory. The local ``finish`` tool records the
agent's own conclusion in ``PlannerResult.finish_result``. That conclusion is
not a benchmark success signal; always use the environment's score or success
predicate for evaluation.

RoboDojo example
----------------

The EEF example in ``examples/robodojo/eef_episode_policy.py`` runs RoboProbe's
Cartesian action and CuRobo planning logic through a single RPent episode.
Its MCP ``move_eef`` result contains the post-action robot state and three RGB
cameras. ``install_eef_episode.py`` places this user-owned policy beside the
reference policy in a fresh RoboDojo workspace; ``configure_eef_experiment.py``
adapts the existing parallel launcher. ``cache_gate_eef.py`` replays recorded
RGB observations through the same RPent request path and requires a token
weighted provider cache hit rate above 60% before benchmark submission.
RoboDojo supplies the native task score.

The earlier discrete-action example remains available:

``examples/robodojo`` contains an XPolicyLab bridge that uses ``EmbodiedAgent``
for RoboDojo decisions. Its ``snapshot`` MCP tool supplies camera frames and
state; ``submit_action`` returns short discrete commands. RoboDawn's motion
controller executes those commands, and RoboDojo supplies the native score.
The example pins the RoboDawn controller revision, keeps credentials outside
Git, and includes a launcher for parallel GPU workers. See the example's
``README.md`` for the installation-specific setup and result paths.
