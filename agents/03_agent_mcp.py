'''
Version 3: Agent + local MCP calculator tool + MLflow tracing.

Combines everything: Kimi K2.5 via Ollama reads the image, the MCP server
does the arithmetic, and MLflow records the full trace including tool calls.
'''
#%% [markdown]
# # Version 3: Agent + Local MCP Tool + MLflow Tracing
#
# What changes compared to Version 2:
#
# - A **local MCP server** (`calculator_server.py`) is started automatically
#   as a subprocess via `MCPServerStdio`
# - The agent gains a `calculate` tool it can call at will
# - The LLM decides *when* to call the tool — we don't hardcode it
# - MLflow traces the full run including **tool call spans**
#
# Why use MCP instead of a plain `@agent.tool`?
#
# - MCP servers are language-agnostic and reusable across frameworks
# - They run in isolation (separate process / even separate machine)
# - This mirrors real-world agentic architectures

#%% [markdown]
# ## Imports

import asyncio
import sys
from pathlib import Path

import mlflow
import mlflow.pydantic_ai

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


#%% [markdown]
# ## MLflow Setup

mlflow.pydantic_ai.autolog(log_traces=True)
mlflow.set_experiment("ocr-math-agent-mcp")


#%% [markdown]
# ## MCP Server Connection
#
# `MCPServerStdio` spawns `calculator_server.py` as a child process and
# communicates with it over stdin/stdout using the MCP protocol.
# PydanticAI handles the lifecycle (start, stop) automatically.

calculator_server = MCPServerStdio(
    command=sys.executable,              # use the same Python interpreter
    args=["calculator_server.py"],
    cwd=Path(__file__).parent,           # look for the server in the same folder
)

#%% [markdown]
# ## Model & Agent
#
# The only difference from Version 1 is `toolsets=[calculator_server]`.
# PydanticAI fetches the tool list from the MCP server at startup and
# exposes them to the model automatically.

provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

model = OpenAIChatModel("kimi-k2.5:cloud", provider=provider)

agent = Agent(
    model,
    toolsets=[calculator_server],        # <-- the only new line vs. Version 1
    system_prompt=(
        "You are an expert OCR and math assistant. "
        "When you receive an image, do two things:\n"
        "1. Extract ALL visible text exactly as it appears.\n"
        "2. If you find any mathematical expressions or equations, "
        "use the 'calculate' tool to solve them — do NOT calculate in your head. "
        "Show the expression and the result clearly.\n"
        "Keep your response concise and well structured."
    ),
)

#%% [markdown]
# ## Image Analysis Function
#
# MLflow traces the full run including the MCP tool call spans,
# so you can see exactly when `calculate` was invoked and with what arguments.

async def analyze_image(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()

    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    with mlflow.start_run(run_name="analyze_image_mcp"):
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("model", "kimi-k2.5:cloud")
        mlflow.log_param("mcp_server", "calculator_server.py")

        # The agent will call the MCP `calculate` tool for any math it finds
        result = await agent.run(
            [
                "Please read this image. Extract all text and solve any math you find.",
                BinaryContent(data=image_bytes, media_type=media_type),
            ]
        )

        mlflow.log_text(result.output, "response.txt")

    return result.output


#%% [markdown]
# ## Entry Point

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.png"

    print(f"Analyzing image: {image_path}")
    print("MCP server: calculator_server.py (started automatically)")
    print("MLflow UI:  run `mlflow ui` and open http://127.0.0.1:5000\n")

    response = asyncio.run(analyze_image(image_path))
    print("=" * 50)
    print(response)
