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

"""Configure a fresh RoboProbe workspace for the RPent EEF user demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from install_eef_episode import POLICY, install

NAMESPACE = "0_ckpt_name=rpent-eef-v1,action_type=joint"


def _replace(path: Path, before: str, after: str) -> None:
    content = path.read_text(encoding="utf-8")
    if before not in content:
        raise ValueError(f"expected text missing from {path}: {before}")
    path.write_text(content.replace(before, after), encoding="utf-8")


def configure(experiment: Path, repo: Path, key_file: Path, deps: Path) -> None:
    if not (repo / "rpent/embodied_agent.py").is_file():
        raise FileNotFoundError("RPent checkout is missing")
    if not (deps / "mcp").is_dir() or not (deps / "pydantic_ai").is_dir():
        raise FileNotFoundError("RPent runtime dependencies are missing")
    if not key_file.is_file():
        raise FileNotFoundError("model key file is missing")
    install(experiment)
    manager = experiment / "manage.py"
    _replace(
        manager, 'POLICY = "RoboDojo_Agent_L3_Inspect_EEF"', f'POLICY = "{POLICY}"'
    )
    _replace(
        manager,
        'NAMESPACE = "0_ckpt_name=tokenhub-once,action_type=joint"',
        f'NAMESPACE = "{NAMESPACE}"',
    )
    _replace(
        manager,
        '"ROBODOJO_NUM_ENVS": "1", "ROBODOJO_CKPT": "tokenhub-once"',
        '"ROBODOJO_NUM_ENVS": "1", "ROBODOJO_CKPT": "rpent-eef-v1"',
    )
    _replace(
        manager,
        '"L3_INSPECT_ADDITIONAL_INFO": "ckpt_name=tokenhub-once,action_type=joint"',
        '"L3_INSPECT_ADDITIONAL_INFO": "ckpt_name=rpent-eef-v1,action_type=joint"',
    )
    _replace(
        manager,
        '"RoboDojo", task, "tokenhub-once",',
        '"RoboDojo", task, "rpent-eef-v1",',
    )
    _replace(manager, '"rp-once-{}-s{:02d}-{}-a{}"', '"rpent-eef-{}-s{:02d}-{}-a{}"')
    _replace(manager, '"rp-once-{}-s{:02d}"', '"rpent-eef-{}-s{:02d}"')

    worker = experiment / "worker.sh"
    _replace(
        worker,
        'export PYTHONPATH="${RP_ONCE_DEPS}:${WORKSPACE}',
        f'export PYTHONPATH="{deps}:{repo}:${{RP_ONCE_DEPS}}:${{WORKSPACE}}',
    )
    # The source launcher shares its XDG cache between all workers. Warp can
    # read a partially compiled module when several shards start together.
    # Give each shard its own kernel cache while retaining shared dependencies.
    cache_root = "${CACHE_ROOT}"
    _replace(
        worker,
        'export XDG_CACHE_HOME="${RP_ONCE_CACHE_ROOT}/xdg/isaaclab232"',
        'PLAN_DIR="${SHARD%/*}"\n'
        'PLAN_NAME="${PLAN_DIR##*/}"\n'
        'CACHE_ROOT="${EXPERIMENT}/runtime/worker-cache/${PLAN_NAME}/${SHARD##*/}"\n'
        f'export XDG_CACHE_HOME="{cache_root}/xdg"\n'
        f'export WARP_CACHE_PATH="{cache_root}/warp"',
    )
    _replace(
        worker,
        'export TORCH_EXTENSIONS_DIR="${RP_ONCE_CACHE_ROOT}/torch_extensions/${GPU_TAG}-isaaclab232"',
        f'export TORCH_EXTENSIONS_DIR="{cache_root}/torch_extensions/${{GPU_TAG}}"',
    )
    _replace(
        worker,
        'export CUDA_CACHE_PATH="${RP_ONCE_CACHE_ROOT}/cuda_cache/driver${DRIVER_TAG}"',
        f'export CUDA_CACHE_PATH="{cache_root}/cuda_cache/driver${{DRIVER_TAG}}"',
    )
    _replace(
        worker,
        "from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.model import Model",
        f"from XPolicyLab.policy.{POLICY}.model import Model\n"
        f"from XPolicyLab.policy.{POLICY}.policy import EmbodiedEefPolicy",
    )
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy"] = POLICY
    manifest["namespace"] = NAMESPACE
    manifest["rpent_checkout"] = str(repo)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    destination = experiment / "runtime/openai_api_key"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with key_file.open("rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    os.chmod(destination, 0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    args = parser.parse_args()
    configure(
        args.experiment.resolve(),
        args.repo.resolve(),
        args.key_file.resolve(),
        args.deps.resolve(),
    )
