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

"""Configuration for composed API agents, independent of model SDK imports."""

from typing import TYPE_CHECKING

from rpent.runtime.config import RuntimeConfig, SubAgentConfig

if TYPE_CHECKING:
    from rpent.runtime.context_engine import ContextPolicy, summarize_history
    from rpent.runtime.trace import TraceConfig

__all__ = [
    "RuntimeConfig",
    "SubAgentConfig",
    "ContextPolicy",
    "TraceConfig",
    "summarize_history",
]


def __getattr__(name: str):
    # Configuration discovery remains usable without loading the model SDK.
    if name in {"ContextPolicy", "summarize_history"}:
        from rpent.runtime import context_engine

        return getattr(context_engine, name)
    if name == "TraceConfig":
        from rpent.runtime.trace import TraceConfig

        return TraceConfig
    raise AttributeError(name)
