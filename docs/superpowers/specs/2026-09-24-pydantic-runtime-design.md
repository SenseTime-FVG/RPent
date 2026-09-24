# Configured PydanticAI delegation

The user approved implementation after narrowing the scope to missing agent
composition and assembly. Existing context, execution, memory, artifacts,
Dashboard interaction, and tool cancellation retain their current behavior.

## Contract

- Keep `planner="api"` and `Planner.solve` / `PlannerResult` unchanged.
- Add `RuntimeConfig` containing named `SubAgentConfig` records. Each record
  specifies instructions, optional description/model, skill paths, and tool names.
- Accept the configuration through `EmbodiedAgent(runtime=...)`,
  `build_planner(runtime=...)`, and `--runtime-config PATH` for CLI/Dashboard.
  Reject it for other planners before starting robot/MCP resources.
- Resolve file-config skill paths relative to the configuration file; Python
  configuration paths follow normal caller-relative path semantics.
- Reuse `assemble_context` / `load_skill` for child instructions and the existing
  model builders, request retries, tool wrappers, and history-image processor.
- Use the official `pydantic-ai-harness` `SubAgents` capability, with explicit
  children only, no disk discovery and no implicit parent-tool inheritance.
- Child calls have fresh histories and receive the explicit delegation task.
  They never receive the parent's initial query, memory, or conversation by default.
- Initial child tool access is limited to existing `read_image`, `read_text_file`,
  and `list_dir` readers that are present in the supplied tool catalog. Parent
  environment actions and `finish` remain on the parent. Missing/disallowed
  tool names fail during assembly, before any model call.
- Reuse the existing serial Toolkit contract. A shared async gate around the
  existing toolsets serializes their calls across concurrent children and the
  parent; the delegation tool itself does not acquire that gate.
- Child model calls may run concurrently. Usage and usage limits follow the
  SDK's shared accounting. Parent cancellation awaits delegated work through
  the SDK; existing Toolkit cancellation retains ownership of robot cleanup.
- The unconfigured path creates the same agent and retains existing behavior.

## Dependencies and scope

Keep Python `>=3.10,<3.13`. Add the optional `runtime` extra using
`pydantic-ai-harness>=0.34,<0.35`, which requires core `>=2.44.0`; keep the existing
base installation independent of this extra. Unit-test installation includes it.
No new execution loop, session persistence, memory system, tool-effect metadata,
context budget/compaction system, dynamic skill loader, background task registry,
or dashboard presentation is part of this change.

## Validation

Use offline FunctionModel tests for explicit context/tool isolation, child model
selection, repeated and parallel delegations, shared-tool serialization, failure
and cancellation cleanup, and usage limits. Exercise configuration forwarding
through Python, CLI continuation, and Dashboard entry points. Run existing API
contracts, the full CPU unit suite, pre-commit, both Sphinx builds, and startup
checks. Real models, simulators, and robot hardware are outside this validation.
