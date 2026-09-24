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

"""Run the RPent-backed RoboProbe EEF policy in RoboDojo's native harness."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect import deploy as joint_deploy
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure

from .docs import eef_docs_from_joint_docs
from .policy import EmbodiedEefPolicy


def _planner(task_env: Any):
    manager = getattr(task_env, "robot_manager", None)
    if manager is None:
        raise InfrastructureFailure(
            "RoboDojo robot_manager is required for EEF planning"
        )

    def plan(*, arm: str, target_pose):
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            return {"status": "Unavailable"}
        env_idx = int(getattr(task_env, "env_idx", 0))
        current = manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
        return planner.plan_path(
            current,
            target_pose,
            real_robot_pose=copy.deepcopy(robot.entity_origin_pose),
        )

    return plan


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    active: list[EmbodiedEefPolicy] = []

    def factory(*, action_spec, env, task_env):
        policy = EmbodiedEefPolicy(
            action_spec=replace(
                action_spec, docs=eef_docs_from_joint_docs(action_spec.docs)
            ),
            env=env,
            planner=_planner(task_env),
        )
        active.append(policy)
        return policy

    try:
        joint_deploy.eval_one_episode(TASK_ENV, model_client, policy_factory=factory)
    finally:
        for policy in active:
            policy.close()


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise InfrastructureFailure(
            "RoboDojo started multiple environments; use one episode per worker"
        )
    eval_one_episode(TASK_ENV, model_client)
