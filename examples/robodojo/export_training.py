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

"""Export scored RoboDojo EEF runs to the shared agent training format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rpent import __version__ as rpent_version
from rpent.training_dataset import TrainingDataset


def export_experiment(
    experiment: Path,
    output_root: Path,
    *,
    dataset_id: str,
    dataset_version: str,
    split: str,
    seed: int = 0,
    benchmark_version: str | None = None,
    versions: dict[str, str] | None = None,
    exporter_commit: str | None = None,
    successful_only: bool = False,
    seal: bool = False,
) -> list[Path]:
    """Join native scores and full RPent traces for one RoboDojo experiment.

    Args:
        experiment: Scored RoboDojo experiment root.
        output_root: Parent directory for versioned datasets.
        dataset_id: Stable dataset family identifier.
        dataset_version: Immutable release label for this collection.
        split: Explicit train, validation, or test split for this export.
        seed: Seed used by this experiment's benchmark launcher.
        benchmark_version: Optional simulator/task version label.
        versions: Versions captured for the agent, simulator, scorer and prompts.
        exporter_commit: Commit of the code running this exporter, if known.
        successful_only: Export only native successes.
        seal: Verify and freeze the dataset after this export completes.

    Returns:
        Paths to the new standard episode records.
    """
    status = json.loads((experiment / "results/once-status.json").read_text())
    dataset = TrainingDataset.create(
        output_root,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        producer={
            "name": "rpent-robodojo-export",
            "version": rpent_version,
            "git_commit": exporter_commit,
        },
        metadata={"source_experiment": str(experiment.resolve())},
    )
    exported = []
    for native in status["results"]:
        if successful_only and not native["success"]:
            continue
        task_id, run_id = native["task"], native["run_id"]
        if not all(
            re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in (task_id, run_id)
        ):
            raise ValueError("RoboDojo task and run IDs must be safe path components")
        run_dir = experiment / "runtime/traces" / task_id / run_id / "rpent"
        episode = json.loads((run_dir / "episode.json").read_text())
        instruction = episode.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"RoboDojo run is missing its task instruction: {run_dir}")
        exported.append(
            dataset.add_rpent_trace(
                run_dir,
                {
                    "benchmark": "robodojo",
                    "benchmark_version": benchmark_version,
                    "task_id": task_id,
                    "instruction": instruction,
                    "seed": seed,
                    "metadata": {"native_run_id": run_id},
                },
                split=split,
                outcome={
                    "success": native["success"],
                    "score": native["score"],
                    "metadata": {"native_score_source": native["path"]},
                },
                versions=versions,
            )
        )
    if seal:
        dataset.seal()
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-version")
    parser.add_argument("--versions-file", type=Path)
    parser.add_argument("--exporter-commit")
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    versions = (
        json.loads(args.versions_file.read_text(encoding="utf-8"))
        if args.versions_file is not None
        else None
    )
    for path in export_experiment(
        args.experiment.resolve(),
        args.output_root.resolve(),
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        split=args.split,
        seed=args.seed,
        benchmark_version=args.benchmark_version,
        versions=versions,
        exporter_commit=args.exporter_commit,
        successful_only=args.successful_only,
        seal=args.seal,
    ):
        print(path)
