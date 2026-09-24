# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark-neutral episode runner for user-defined embodied agents."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rpent.embodied_agent import EmbodiedAgent
from rpent.evaluation.result import write_json_atomic


@dataclass(frozen=True, slots=True)
class NativeScore:
    """A score read from the benchmark's own episode result."""

    success: bool
    score: float
    source: str | Path | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("native success must be a bool")
        if not math.isfinite(self.score):
            raise ValueError("native score must be finite")


class BenchmarkEpisode(Protocol):
    """One isolated benchmark episode; it owns reset, execution, and scoring."""

    def run(self, agent: EmbodiedAgent, output_dir: Path) -> NativeScore:
        """Execute and return the benchmark's native score."""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One agent, benchmark, and task combination to evaluate once."""

    agent_name: str
    benchmark_name: str
    task_name: str
    make_agent: Callable[[Path], EmbodiedAgent]
    make_episode: Callable[[], BenchmarkEpisode]

    @property
    def key(self) -> str:
        return f"{self.benchmark_name}/{self.agent_name}/{self.task_name}"


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    output_dir: str | Path,
    *,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run independent cases concurrently and save one native result per case.

    ``make_episode`` must create an isolated environment for each case. GPU
    benchmarks should use a process or cluster worker per case; a shared Isaac
    environment must not be driven by multiple Python threads.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    keys = [case.key for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation cases must have unique keys")
    if any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", part)
        for key in keys
        for part in key.split("/")
    ):
        raise ValueError("case names must contain only letters, digits, _ or -")
    root = Path(output_dir)

    def one(case: EvaluationCase) -> dict[str, Any]:
        destination = root / case.key
        destination.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        try:
            agent = case.make_agent(destination)
            score = case.make_episode().run(agent, destination)
            record: dict[str, Any] = {
                "case": case.key,
                "success": score.success,
                "score": score.score,
                "native_score_source": str(score.source) if score.source else None,
                "details": dict(score.details),
                "elapsed_s": time.monotonic() - started,
                "error": None,
            }
        except Exception as exc:
            record = {
                "case": case.key,
                "success": None,
                "score": None,
                "native_score_source": None,
                "details": {},
                "elapsed_s": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_json_atomic(destination / "result.json", record)
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(one, case): case.key for case in cases}
        records = {futures[future]: future.result() for future in as_completed(futures)}
    return [records[key] for key in keys]
