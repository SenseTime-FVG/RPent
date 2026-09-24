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

"""Expose the user policy's EEF tool schemas over MCP.

Motion calls are handed to the RoboDojo process by EmbodiedEpisode. This
server only advertises the exact schemas that the policy supplies.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool


async def serve(schema_path: Path) -> None:
    specs = json.loads(schema_path.read_text(encoding="utf-8"))
    server = Server("robodojo-eef")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["parameters"],
            )
            for item in specs
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        raise RuntimeError(f"{name} must be executed by the RoboDojo episode")

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1])))
