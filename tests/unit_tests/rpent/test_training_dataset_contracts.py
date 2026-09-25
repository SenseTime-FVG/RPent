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

"""Dataset releases are indexed, reproducible, and task-disjoint."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rpent.training_dataset import TrainingDataset


def _task(task_id: str = "put/block") -> dict:
    return {
        "benchmark": "robodojo",
        "benchmark_version": "2026-09",
        "task_id": task_id,
        "instruction": "Put the block in the bowl",
        "seed": 0,
    }


def _calls(image: bytes = b"camera") -> list[dict]:
    return [
        {
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            "Put the block in the bowl",
                            {"type": "image", "media_type": "image/png", "data": image},
                        ],
                    }
                ]
            },
            "response": "done",
        }
    ]


def test_dataset_manifest_indexes_reopen_seal_and_verifies_assets(
    tmp_path: Path,
) -> None:
    dataset = TrainingDataset.create(
        tmp_path,
        dataset_id="robot-actions",
        dataset_version="2026-09.v1",
        producer={"name": "collection-job", "version": "2", "git_commit": "abc123"},
        metadata={"license": "internal"},
    )
    first = dataset.add_episode(
        _task(),
        _calls(),
        episode_id="run/1",
        split="train",
        outcome={"success": True, "score": 1.0},
        provenance={
            "source_run_id": "native-run-1",
            "versions": {"agent_commit": "abc123"},
        },
    )
    second = dataset.add_episode(
        _task(), _calls(b"other"), episode_id="run-2", split="train"
    )
    assert first.parent != second.parent
    assert first.parent.parent.name == "put%2Fblock"
    manifest = TrainingDataset.open(dataset.root).verify()
    assert manifest["dataset_version"] == "2026-09.v1"
    assert manifest["producer"]["git_commit"] == "abc123"
    assert manifest["metadata"]["license"] == "internal"
    assert [item["split"] for item in manifest["episodes"]] == ["train", "train"]
    assert all(len(item["sha256"]) == 64 for item in manifest["episodes"])
    assert dataset.seal()["status"] == "sealed"
    with pytest.raises(ValueError, match="sealed dataset"):
        dataset.add_episode(_task("new"), _calls(), episode_id="new", split="train")

    image = next((first.parent / "assets").iterdir())
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.verify()


def test_dataset_rejects_split_leakage_and_duplicate_episodes(tmp_path: Path) -> None:
    dataset = TrainingDataset.create(
        tmp_path,
        dataset_id="actions",
        dataset_version="v1",
        producer={"name": "test", "version": "1"},
    )
    with pytest.raises(ValueError, match="empty dataset"):
        dataset.seal()
    dataset.add_episode(_task(), _calls(), episode_id="one", split="train")
    with pytest.raises(ValueError, match="multiple splits"):
        dataset.add_episode(_task(), _calls(), episode_id="two", split="test")
    with pytest.raises(FileExistsError, match="already exists"):
        dataset.add_episode(_task(), _calls(), episode_id="one", split="train")
    assert len(dataset.verify()["episodes"]) == 1


def test_dataset_manifest_detects_modified_episode(tmp_path: Path) -> None:
    dataset = TrainingDataset.create(
        tmp_path,
        dataset_id="actions",
        dataset_version="v1",
        producer={"name": "test", "version": "1"},
    )
    document = dataset.add_episode(_task(), _calls(), episode_id="one", split="train")
    document.write_text(document.read_text() + " ")
    with pytest.raises(ValueError, match="episode checksum mismatch"):
        dataset.verify()


def test_dataset_serializes_parallel_writers_and_rejects_orphans(
    tmp_path: Path,
) -> None:
    dataset = TrainingDataset.create(
        tmp_path,
        dataset_id="actions",
        dataset_version="v1",
        producer={"name": "test", "version": "1"},
    )

    def add(index: int) -> Path:
        return dataset.add_episode(
            _task(f"task-{index}"), _calls(), episode_id=f"run-{index}", split="train"
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        documents = list(pool.map(add, range(4)))
    assert len(documents) == len(dataset.verify()["episodes"]) == 4
    orphan = dataset.root / "episodes/train/robodojo/orphan/run/episode.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}")
    with pytest.raises(ValueError, match="unindexed"):
        dataset.seal()
