# Pydantic Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan
> task-by-task. The user explicitly authorized implementation in this task.

**Goal:** Configure isolated PydanticAI delegates while reusing RPent's API planner.

**Architecture:** Two new modules own declarative configuration and agent assembly.
The existing API planner drives the assembled agent. The official SubAgents
capability owns delegation, and existing RPent components retain their contracts.

**Tech Stack:** Python, PydanticAI, pydantic-ai-harness 0.34, pytest FunctionModel.

**Spec:** `docs/superpowers/specs/2026-09-24-pydantic-runtime-design.md`

## Global Constraints

- Keep Python `>=3.10,<3.13`.
- Keep `planner="api"` and `Planner.solve` / `PlannerResult` unchanged.
- Optional harness dependency: `pydantic-ai-harness>=0.34,<0.35`.
- Reuse existing modules without adding context, memory, or artifact features.
- Unit tests are offline and use ordinary CPU hardware.

## Review Focus

- Relative skill paths must work when the process cwd differs from the config directory.
- Child calls must not expose parent memory, robot actions, or `finish`.
- Concurrent delegates sharing readers must not trip Toolkit's single-operation guard.
- Timeout, cancellation, and sibling failure must leave no live delegated coroutine.
- Empty configuration must preserve single-agent behavior and lightweight imports.

## Task 1: Configuration and assembly

Files: create `rpent/runtime/{__init__,config,factory}.py`,
`tests/unit_tests/rpent/runtime/test_runtime_contracts.py`; update `pyproject.toml`.

Interfaces: `SubAgentConfig(instructions, description=None, model=None, skills=(),
tools=())`, `RuntimeConfig(subagents={})`, `RuntimeConfig.from_file(path)`,
`build_runtime_agent(model=..., system_prompt=..., tools=..., max_tokens=...,
capabilities=..., runtime=...) -> Agent`.

- [x] Write failing tests that exercise isolated, named delegation:
  ```python
  config = RuntimeConfig(subagents={"reviewer": SubAgentConfig(instructions="Review.")})
  result = asyncio.run(agent.run("Private root query"))
  assert result.output == "review accepted"
  assert child_prompts == ["Review this explicit task"]
  ```
- [x] Run `pytest tests/unit_tests/rpent/runtime/test_runtime_contracts.py -q`;
  confirm missing runtime imports fail inside test bodies.
- [x] Implement configuration validation, relative-path resolution, assembly,
  official delegation, reader filtering and shared toolset serialization.
- [x] Add observable tests for custom models, skill freshness, unavailable readers,
  serialized shared reads, repeated/parallel delegation, and cancellation cleanup.
- [x] Run focused tests until all pass.

## Task 2: Existing planner and entry points

Files: modify `rpent/planner/{api_loop,base}.py`, `rpent/embodied_agent.py`,
`rpent/cli/{main,dashboard}.py`; extend their existing contract tests.

Interfaces: optional `runtime: RuntimeConfig | None` on the API planner builder
and EmbodiedAgent, plus CLI `--runtime-config PATH`.

- [x] Add failing planner integration coverage using FunctionModel:
  ```python
  result = planner.solve(system_prompt="Act.", user_message="Task",
                         toolkit=toolkit, max_turns=10)
  assert result.finish_result["status"] == "success"
  assert result.stats["requests"] == 3  # parent, child, parent finish
  ```
- [x] Add tests for early non-api rejection and configuration reaching CLI
  continuation/Dashboard and the Python MCP entry point.
- [x] Wire the new assembly into `_build_agent`, preserving existing run loops.
- [x] Verify usage-limit exits preserve usage and root timeouts drain children.
- [x] Run runtime, planner, CLI, Dashboard and EmbodiedAgent focused tests.

## Task 3: Documentation and validation

Files: paired `usage/configure_planner.rst` and `usage/embodied_agent.rst`.

- [x] Document optional installation, Python/YAML examples, reader restrictions,
  model inheritance/override, path handling, isolation and SDK usage semantics.
- [x] Run `pre-commit run --all-files` and `pytest tests/unit_tests -v`.
- [x] Run `sphinx-build -W --keep-going docs/source-en docs/build/html-en`
  and the equivalent Chinese build.
- [x] Review the complete diff and record exact results and unverified paths.

## Execution record

- Branch `codex/pydantic-runtime` starts from the existing context change rebased
  onto `origin/main` at `2cb072a`.
- Task 1: 21 behavior tests failed before the runtime module existed, then passed.
- Task 2: planner/CLI tests failed on missing configuration parameters, and
  Python/Dashboard/continuation tests failed before forwarding was implemented.
- Ruling: SubAgents 0.34 forwards usage but supplies no parent usage limits to
  child runs, contrary to the documented whole-tree limit behavior. A regression
  reproduced 50 requests with a parent cap of 3. A small run hook now forwards
  the existing request limit; it adds no budget policy or accounting system.
- Runtime and API planner contracts: 42 passed after the limit forwarding fix.
- Final validation: Ubuntu/WSL, Python 3.11.15, pydantic-ai-slim 2.49.0,
  pydantic-ai-harness 0.34.0, existing CPU Flywheel dependencies. Used a Linux
  archive of HEAD plus all 20 working-diff files, copied byte-for-byte, to retain
  LF endings on unchanged shell scripts.
- `pre-commit run --all-files`: passed after reviewing formatter/import fixes.
- `pytest tests/unit_tests -v --junitxml=unit-results.xml`: **682 passed, 1 skipped**
  in 48.64 seconds. The skip is the optional RLinf RoboTwin seed-language check.
- Both `sphinx-build -W --keep-going docs/source-{en,zh} docs/build/html-{en,zh}`
  builds passed. Full logs and JUnit are in `/tmp/rpent-runtime-check-tcc47jk0`.
- Optional-import startup probe: with harness imports blocked, absent/empty runtime
  configurations run normally; configured delegates give installation guidance.
- Independent read-only review found no implementation defect. Its P2 documentation
  finding was fixed in both languages: concurrently starting SDK requests can
  overshoot the recorded-usage threshold. No custom hard-budget accounting was added.
- No live model, simulator, GPU policy-chain, or real-robot run was performed.
