# Agent runtime examples

Run these commands from the repository root with Python 3.10–3.12:

```bash
pip install -e ".[test,runtime]" imageio imageio-ffmpeg
python -m examples.runtime.offline_demo --output /tmp/rpent-runtime-demo-001
python -m examples.runtime.offline_demo --output /tmp/rpent-runtime-summary-001 --compress
```

Use a new or empty directory for each run. No API key, model server, GPU, robot,
or network service is used by the demo. `FunctionModel` supplies scripted replies
and **synthetic token/cache counts**; NumPy/imageio generate synthetic frames and
video. Those numbers are fixture data, not model performance measurements.

The demo runs the real runtime factory, per-request context policy, on-demand
skill tools, delegated child, retry wrapper, trace recorder and HTML exporter.
It reads `operation/SKILL.md` and its referenced checklist, calls observation and
movement tools, and asks a child with its own `review` skill to inspect explicitly
supplied evidence. The child cannot read the parent's skill catalog.

Open `RUN_DIR/report/index.html` directly in a browser. The report contains the
actual requests sent to the scripted model, tool inputs/results, child calls,
synthetic video, and token/cache statistics. Move the whole `report/` directory
when sharing it. To export the same run again into another empty directory:

```bash
python -m rpent.cli.trajectory /tmp/rpent-runtime-demo-001 \
  --output /tmp/rpent-runtime-report-001
```

`custom_context.py` provides a stateful async factory that summarizes discarded
history while retaining the current task and complete recent tool exchanges.
The `--compress` run makes extra summary requests through `summarize_history`.
Their usage consumes the same request budget and appears in the report as
`context_compression`; compression is not free and need not improve caching.
The ordinary run uses `recent_turns` with two retained response/feedback groups.

`runtime.yaml` is a real-provider configuration, separate from the offline demo.
Replace `YOUR_MODEL`, set the provider's normal environment credentials, and
pass it to a configured robot entry point:

```bash
rpent --robot libero --planner api --runtime-config examples/runtime/runtime.yaml \
  --suite libero_goal_task --task 1 --no-vla
```

The robot command requires that robot's environment and task assets. `--no-vla`
skips VLA components and tools; it does not remove simulator or planner model
requirements. Omit it to use the integration's normal VLA startup. Do not also
pass `--model`/`--base-url` when the runtime file declares `llm`. YAML paths resolve
relative to the configuration file. The runtime defaults to full trace capture;
set `trace.mode` to `metadata` or `off` when full inputs/outputs should not be saved.

Complete documentation: [English](../../docs/source-en/rst_source/usage/agent_runtime.rst)
and [中文](../../docs/source-zh/rst_source/usage/agent_runtime.rst).
