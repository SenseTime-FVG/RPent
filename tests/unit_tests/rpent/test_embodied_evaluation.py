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

"""The generic runner keeps native scoring and episode isolation intact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.evaluation.embodied import EvaluationCase, NativeScore, evaluate_cases


class _Benchmark:
    def __init__(self, success: bool) -> None:
        self.success = success

    def run(self, agent, output_dir: Path) -> NativeScore:
        assert agent == output_dir
        native = output_dir / "native.json"
        native.write_text(json.dumps({"success": self.success}), encoding="utf-8")
        return NativeScore(self.success, 1.0 if self.success else 0.25, native)


def test_evaluate_cases_keeps_native_score_and_isolated_outputs(tmp_path: Path) -> None:
    cases = [
        EvaluationCase(
            "agent",
            "benchmark",
            f"task_{index}",
            make_agent=lambda path: path,
            make_episode=lambda success=success: _Benchmark(success),
        )
        for index, success in enumerate((True, False))
    ]
    records = evaluate_cases(cases, tmp_path, max_workers=2)
    assert [record["success"] for record in records] == [True, False]
    assert [record["score"] for record in records] == [1.0, 0.25]
    assert all(Path(record["native_score_source"]).is_file() for record in records)
    with pytest.raises(ValueError, match="unique keys"):
        evaluate_cases([cases[0], cases[0]], tmp_path / "duplicate")
