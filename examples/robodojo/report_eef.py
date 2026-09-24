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

"""Join RoboDojo native scores with RPent usage for scored EEF episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpent.evaluation.result import write_json_atomic


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def report(experiment: Path) -> dict:
    status = json.loads((experiment / "results/once-status.json").read_text())
    rows = []
    for native in status["results"]:
        task = native["task"]
        run_id = native["run_id"]
        root = experiment / "runtime/traces" / task / run_id / "rpent"
        requests = root / "llm_requests.jsonl"
        attempts = (
            [json.loads(line) for line in requests.read_text().splitlines() if line]
            if requests.is_file()
            else []
        )
        successes = [item for item in attempts if item.get("outcome") == "success"]
        usage = {
            field: sum(
                int((item.get("usage") or {}).get(field) or 0) for item in successes
            )
            for field in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            )
        }
        episode_path = root / "episode.json"
        episode = json.loads(episode_path.read_text()) if episode_path.is_file() else {}
        rows.append(
            {
                "task": task,
                "run_id": run_id,
                "native_success": native["success"],
                "native_score": native["score"],
                "native_score_path": native["path"],
                "llm_requests": len(successes),
                "failed_attempts": sum(
                    item.get("outcome") == "error" for item in attempts
                ),
                "http_500_attempts": sum(
                    item.get("status_code") == 500 for item in attempts
                ),
                "agent_elapsed_s": episode.get("agent_elapsed_s"),
                "agent_error": episode.get("agent_error"),
                **usage,
                "cache_hit_rate": (
                    usage["cache_read_tokens"] / usage["input_tokens"]
                    if usage["input_tokens"]
                    else None
                ),
            }
        )
    calls = [row["llm_requests"] for row in rows if row["llm_requests"]]
    input_tokens = sum(row["input_tokens"] for row in rows)
    cache_read = sum(row["cache_read_tokens"] for row in rows)
    return {
        "scored": len(rows),
        "total": status["total"],
        "native_success": sum(row["native_success"] for row in rows),
        "native_score_sum": sum(row["native_score"] for row in rows),
        "llm_requests_total": sum(calls),
        "llm_requests_min": min(calls) if calls else None,
        "llm_requests_median": _percentile(calls, 0.5),
        "llm_requests_p75": _percentile(calls, 0.75),
        "llm_requests_max": max(calls) if calls else None,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_hit_rate": cache_read / input_tokens if input_tokens else None,
        "tasks": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.experiment.resolve())
    destination = args.experiment / "results/rpent-eef-report.json"
    write_json_atomic(destination, result)
    print(
        f"scored={result['scored']}/{result['total']} "
        f"success={result['native_success']} "
        f"cache_hit_rate={result['cache_hit_rate']} "
        f"report={destination}"
    )
