# Agent runtime and trajectory implementation plan

> Execute task by task, testing each completed step. The user approved implementation, upstream integration, commit and push, and requested complete documentation and examples.

**Goal:** Deliver configurable per-request context and skills, optional VLA, bounded model retries, and shared live/offline trajectory inspection.

**Architecture:** Extend the existing API planner and runtime factory. Keep input preparation with the caller, reuse model/robot ownership, and persist runtime events for one shared trajectory projection.

**Tech Stack:** Python 3.10–3.12, PydanticAI, existing FastAPI/SSE Dashboard, plain JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-agent-runtime-context-trajectory-design.md`.

## Global constraints

- Callers already organize their inputs into prompts; no benchmark comparison or new benchmark adapter layer.
- Keep existing PlannerResult, cancellation, robot ownership, SDK child isolation and legacy log contracts.
- Three retries means initial request plus at most three retries. Never replay executed tools.
- Parent, child and compression usage count once; prompt-cache tokens are part of input tokens.
- First release includes both live Dashboard and offline HTML, actual model input/output, media and token/cache statistics.
- Optional simulator/model/harness imports stay optional. Preserve existing working-tree changes.
- Each worker owns explicit files, tests its changes, and never commits or reverts another worker's changes.

## Review focus

- First-turn pruning must not destroy the original observation in the trace.
- Concurrent/repeated delegates must not share mutable context state or expose another agent's skills.
- Partial model failures must retain reported usage without counting cumulative snapshots twice.
- VLA disabled must skip loading/connecting components as well as hiding tool schemas.
- Trace rendering must handle incomplete streams and missing media, and safely display untrusted text.

## Task 1: Baseline, upstream and data conversion

Files: existing runtime work, `rpent/context.py`, `rpent/data_convert.py`, `rpent/runtime/skills.py`, consumers in CLI/EmbodiedAgent/runtime, context tests.

- [x] Run the current runtime/API/LLM contracts in a supported Python environment.
- [x] Preserve tested runtime work in a commit, fetch/integrate latest origin/main, resolve conflicts retaining both sets of contracts, retest affected entry points.
- [x] Rename ContextDocument/ContextBundle/assemble_context to TextDocument/PlannerInput/convert_planner_input; move file loading to runtime.skills. Preserve rendered output and initial_context argument.
- [x] Run converted input and entry-point tests before proceeding.

## Task 2: Context, skills and model configuration

Files: `rpent/runtime/{config,factory,context_engine,skills,__init__}.py`, relevant runtime/LLM tests. Root owns forwarding in planner and entry points.

Interfaces: RuntimeConfig accepts context_engine or context_engine_factory, built-in context policy, skill_paths, llm, trace; SubAgentConfig supports corresponding isolated configuration. `read_skill(name, resource="SKILL.md")`; `summarize_history(ctx, messages, *, llm=None)` shares ctx usage/limits.

- [x] Add failing behavior tests for per-request sync/async history processing, valid tool groups, child state isolation and shared summary budget.
- [x] Implement callbacks via current SDK capability boundary; image pruning remains the final existing policy.
- [x] Add skill tests for lazy read, fresh resource content, per-child catalogs, traversal/size errors, and preserved eager skills.
- [x] Reuse LLMConfig for root/child full configuration, preserve model shorthand, and reject conflicting explicit settings.
- [x] Run runtime contracts and root integration contracts; record results.

## Task 3: Runtime events and request recording

Files: new `rpent/runtime/events.py` and `trace.py`, `rpent/llm/retry.py`, `rpent/planner/api_loop.py`, root entry points, focused tests.

Interfaces: `TraceConfig`, run-owned `TraceRecorder`, scoped `current_trace`, `RuntimeEvent`; recorder emits envelope with seq/type/run/agent/turn/request/attempt/tool IDs. Snapshot references point into trace directory; legacy request logs retain sanitized shape/usage only.

- [x] Add failing tests for lifecycle closure, retry count, original input retention and snapshot serialization.
- [x] Implement run writer and PydanticAI lifecycle capture for parent/child/tool/context; capture inputs before pruning and final model request after processing.
- [x] Extend existing retry wrapper with correlated attempt events and snapshots, default max_retries=3, preserve partial-stream behavior.
- [x] Wire CLI, Dashboard and EmbodiedAgent ownership through toolkit close/media finalization.
- [x] Run event/trace, retry, planner and entry-point contracts.

## Task 4: Optional VLA

Files: five `robots/{libero,robocasa,robotwin,franka,dual_franka}/` integrations and related tests. Root owns shared CLI/Dashboard switches.

Interfaces: startup `enable_vla=False` / `--no-vla`; toolkit-owned accurate VLA tool sets; skip component startup, accept absent clients and render appropriate prompts.

- [x] Add failing tests proving disabled VLA never starts/connects and schemas/dispatch agree.
- [x] Update constructors, resets and runtime component selection with existing ownership helpers.
- [x] Keep default enabled behavior and report unsupported Flash combinations before resource startup.
- [x] Run affected robot contracts and CLI/Dashboard entry-point tests.

## Task 5: Shared live/offline trajectory views

Files: `rpent/evaluation/trajectory.py`, `rpent/cli/trajectory.py`, Dashboard server/static files, projection/export/UI tests.

Interfaces: `TrajectoryProjection.apply(event)`, run-directory reader; CLI `python -m rpent.cli.trajectory RUN_DIR --output REPORT_DIR`. UI uses shared projection with per-turn loading and media resolution.

- [x] Add synthetic fixtures with retries, child/compression requests, partial lines, missing cache fields and media.
- [x] Implement one usage projection, actual request/response/tool display, timeline and media references.
- [x] Integrate SSE summary/version notifications and on-demand data GET; reuse the same JS component in standalone HTML.
- [x] Ensure export works after moving its whole output directory; unknown/missing/partial states remain explicit.
- [x] Run projection/export/server tests and exercise shared frontend with an offline fixture.

## Task 6: Documentation, examples and delivery

Files: paired runtime/trajectory usage pages, existing relevant guides, `examples/runtime/` runnable offline demonstration and configuration/skill/context examples.

- [x] Document implemented APIs and migration names, retry/usage semantics, trace modes, VLA switch and supported entry points.
- [x] Run the offline example to produce a complete sample trace and export; include commands for real configured models without embedding credentials.
- [x] Run pre-commit, complete CPU unit suite, both Sphinx builds, startup probes and focused browser verification.
- [x] Independently review implementation and fix findings; preserve test evidence and unverified hardware limitations.
- [ ] Fetch/integrate latest main again if changed, rerun affected checks, commit and push the topic branch. Verify the remote commit.

## Execution record

- Initial state: branch codex/pydantic-runtime at 88e2dd0 with existing staged runtime work and a design document. No business code changes discarded.
- Baseline: 61 focused runtime/API/LLM tests passed in WSL Ubuntu/Python 3.11.15. Preserved the existing runtime work in `5072d4f`, then merged main through `6464846` in `2b8c844`; 120 affected contracts passed. A later `git fetch origin main` confirmed the same main revision.
- Context, skills and model configuration: 299 relevant contracts passed; the latest isolated runtime/data-conversion check passed 73 tests. Root model forwarding, configuration conflicts, `--no-vla` and trace ownership have direct entry-point regressions; CLI plus ownership tests passed 35 tests.
- Optional VLA and media: robot suite passed 241 tests with one expected skip. Schema/dispatch and no-start contracts cover five integrations. Exact frame mapping and trajectory regressions passed 30 tests after fixing concurrent cumulative usage, unfinished attempts, explicit media failures and custom tool frame metadata.
- Events, retries and live views: request/trace/LLM/API contracts passed 67 tests. Dashboard plus trace contracts passed 59 tests; the three Node live-adapter tests passed, including stale request rejection and SSE reconnect/coalescing. Artifact/final manifests are visible before their live notification.
- Documentation: both `sphinx-build -b html -W --keep-going docs/source-{en,zh} ...` builds succeeded. Ordinary and asynchronous-compression demos ran end to end, including real factory/skill/delegate/trace paths, synthetic video encoding and offline export. Browser checks confirmed actual request/response details, live SSE loading, media decoding and action-video navigation. Direct `file://` browser automation is blocked by the browser policy; the report's static resources and relative media references are independently covered by export tests.
- Startup/package: a lightweight environment without Torch, simulator or VLA packages passed basic imports, all five robot discoveries, CLI help and trajectory help. Blocking harness imports still allowed a real two-turn single-agent run with context/tools. A built wheel contained every new runtime/trajectory module and HTML/CSS/JS asset, omitted `rpent/context.py`, and successfully exported an existing trace and video when imported from the unpacked wheel.
- Full CPU suite first pass: 812 passed, one skipped, one failed. The failure was the existing RoboCasa exact prompt-variable assertion missing the intentional new `enable_vla=True` field; the corrected config contract suite passed all 19 tests. Independent review found successful early finish and request-limit exits incorrectly classified as SDK cancellation. Explicit stop reasons fixed these outcomes while retaining real user cancellation; runtime/API/Dashboard regressions passed 134 tests. Finalization now preserves failed media and existing tool associations.
- Root model configuration also reaches transcript/result metadata and the Dashboard's model diagnostic. Actual CLI finalizer tests passed both model-selection paths; 124 diagnostic/CLI/Dashboard tests passed, including explicit credentials, custom transports, short timeouts, and redaction before truncation. HTTP requests cannot replace server-owned endpoint credentials.
- Final bilingual Sphinx builds both succeeded with warnings treated as errors. The final exported sample reports are `logs/runtime-report-final-20260924/index.html` and `logs/runtime-summary-report-final-20260924/index.html`; generated run/media/build files remain ignored and are not included in commits.
- Final required checks on 2026-09-25: `pre-commit run --all-files` passed both Ruff hooks; `python -m pytest tests/unit_tests -v` passed **830 tests, with one skip**, in 94.91 seconds. The two warnings come from SDK deprecation of the test transport's `httpx.AsyncClient`; no test failed. The first formatting run normalized one pre-existing quote in `examples/robodojo/configure_eef_experiment.py`; the reviewed formatting-only change is included so repository-wide checks pass.
- Final browser adapter check: `node --test tests/unit_tests/rpent/dashboard/test_trajectory_live.mjs` passed all three tests; both trajectory JavaScript files passed `node --check`. Final `git fetch origin main` and `git merge --no-edit origin/main` confirmed that `6464846` is already integrated.
- Full environment: `/tmp/rpent-runtime-full-20260924`, Python 3.11.15, PydanticAI 2.49.0, harness 0.34.0, Torch 2.7.1 CPU, LeRobot 0.3.3. Installed `.[test,flywheel]`, docs requirements, pre-commit 4.6.2 and imageio-ffmpeg. No live provider, GPU simulator, VLA checkpoint or physical robot was exercised; the Python 3.10/3.12 CI matrix remains unverified locally.
