# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

"""A live cache report must tolerate a decision log being written."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_cache_report_skips_incomplete_decision_file(tmp_path: Path) -> None:
    path = Path(__file__).parents[3] / "examples/robodojo/cache_report.py"
    spec = importlib.util.spec_from_file_location("cache_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    episode = tmp_path / "rp-once-test-s00-task-a1/episode_001"
    episode.mkdir(parents=True)
    (episode / "decision_0001.json").write_text(
        json.dumps(
            {
                "attempts": 2,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 70},
                },
            }
        )
    )
    (episode / "decision_0002.json").write_text('{"usage":')

    report = module.summarize(tmp_path, "rp-once-test-")
    assert report["decisions"] == 1
    assert report["incomplete_decisions"] == 1
    assert report["model_requests"] == 2
    assert report["cache_hit_rate"] == 0.7
