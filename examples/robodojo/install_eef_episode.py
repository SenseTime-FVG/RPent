#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the RPent user demo beside RoboProbe's reference EEF policy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

POLICY = "RoboDojo_EmbodiedAgent_EEF"
REFERENCE = "RoboDojo_Agent_L3_Inspect_EEF"


def install(experiment: Path) -> Path:
    root = experiment / "workspace/XPolicyLab/policy"
    source = root / REFERENCE
    target = root / POLICY
    if not (source / "deploy.py").is_file():
        raise FileNotFoundError("prepare the RoboProbe experiment first")
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    examples = Path(__file__).resolve().parent
    for src, dest in (
        ("eef_episode_policy.py", "policy.py"),
        ("eef_episode_deploy.py", "deploy.py"),
    ):
        shutil.copy2(examples / src, target / dest)
    config = target / "deploy.yml"
    content = config.read_text(encoding="utf-8")
    before = f"policy_name: {REFERENCE}"
    if before not in content:
        raise ValueError("unexpected RoboProbe deploy.yml policy name")
    config.write_text(
        content.replace(before, f"policy_name: {POLICY}", 1), encoding="utf-8"
    )
    for name in (
        "eval.sh",
        "setup_eval_env_client.sh",
        "setup_eval_policy_server.sh",
    ):
        script = target / name
        content = script.read_text(encoding="utf-8")
        content = content.replace(
            f"export XPL_POLICY_NAME={REFERENCE}",
            f"export XPL_POLICY_NAME={POLICY}",
            1,
        ).replace(
            "export L3_INSPECT_TRACE_NAMESPACE=xpolicylab-l3-inspect-eef",
            "export L3_INSPECT_TRACE_NAMESPACE=rpent-embodied-agent-eef",
            1,
        )
        script.write_text(content, encoding="utf-8")
    # The native harness already has the final observation after each played
    # chunk. Hand it to the user's policy before deciding whether to end the
    # episode, without changing behavior for other policy implementations.
    harness = root / "RoboDojo_Agent_L3_Inspect/deploy.py"
    content = harness.read_text(encoding="utf-8")
    anchor = "            policy.confirm_executed(played)\n"
    if content.count(anchor) != 1:
        raise ValueError("unexpected RoboProbe confirm_executed hook")
    content = content.replace(
        anchor,
        anchor
        + "            if hasattr(policy, 'complete_post_action'):\n"
        + "                policy.complete_post_action(\n"
        + "                    executed, episode_done=TASK_ENV.is_episode_end()\n"
        + "                )\n",
        1,
    )
    harness.write_text(content, encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    print(install(args.experiment.resolve()))
