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

"""Explicit sub-agent configuration for the API planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SubAgentConfig:
    """Instructions, model and selected readers for one named delegate.

    An omitted model reuses the parent's configured model, including its
    endpoint and retries. An explicit provider-prefixed model uses that
    provider's environment configuration. Skill files are read per episode.
    """

    instructions: str
    description: str | None = None
    model: str | None = None
    skills: tuple[str | Path, ...] = ()
    tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("sub-agent instructions must be non-empty text")
        for name, value in (("description", self.description), ("model", self.model)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"sub-agent {name} must be non-empty text")
        if not isinstance(self.skills, (list, tuple)) or any(
            not isinstance(path, (str, Path)) or not str(path).strip()
            for path in self.skills
        ):
            raise ValueError("sub-agent skills must be a sequence of file paths")
        if not isinstance(self.tools, (list, tuple)) or any(
            not isinstance(name, str) or not name.strip() for name in self.tools
        ):
            raise ValueError("sub-agent tools must be a sequence of tool names")
        if len(self.tools) != len(set(self.tools)):
            raise ValueError("sub-agent tool names must be unique")
        object.__setattr__(self, "skills", tuple(Path(path) for path in self.skills))
        object.__setattr__(self, "tools", tuple(self.tools))


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Named delegates exposed by one API planner; empty means no delegation."""

    subagents: Mapping[str, SubAgentConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subagents, Mapping):
            raise ValueError("subagents must map names to SubAgentConfig values")
        for name, config in self.subagents.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(
                    "sub-agent names must be non-empty without surrounding whitespace"
                )
            if not isinstance(config, SubAgentConfig):
                raise ValueError(f"sub-agent {name!r} requires a SubAgentConfig")
        object.__setattr__(self, "subagents", dict(self.subagents))

    @classmethod
    def from_file(cls, path: str | Path) -> RuntimeConfig:
        """Load YAML or JSON, resolving skill paths against the config directory.

        File errors propagate to the caller.

        Args:
            path: Configuration file containing a ``subagents`` mapping.

        Returns:
            Validated configuration with absolute skill paths.

        Raises:
            ValueError: If configuration fields or values are invalid.
        """
        import yaml

        path = Path(path).resolve()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid runtime config {path}: {exc}") from exc
        if not isinstance(data, dict) or set(data) - {"subagents"}:
            raise ValueError("runtime config must be a mapping with only 'subagents'")
        definitions = data.get("subagents", {})
        if not isinstance(definitions, dict):
            raise ValueError("subagents must be a mapping")
        agents = {}
        for name, values in definitions.items():
            if not isinstance(values, dict):
                raise ValueError(f"sub-agent {name!r} must be a mapping")
            try:
                config = SubAgentConfig(**values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid sub-agent {name!r}: {exc}") from exc
            agents[name] = replace(
                config,
                skills=tuple(
                    (path.parent / skill).resolve() for skill in config.skills
                ),
            )
        return cls(subagents=agents)
