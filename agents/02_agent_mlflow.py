'''
Version 2: Same agent as 01_agent.py, now with MLflow tracing.

MLflow records every agent run as a trace — including the model call,
token usage, inputs, and outputs — visible in the MLflow UI.
'''
#%% [markdown]
# # Version 2: Agent + MLflow Tracing
#
# What changes compared to Version 1:
#
# - `mlflow.pydantic_ai.autolog()` patches PydanticAI automatically
# - Every `agent.run()` becomes a **MLflow trace** with nested spans
# - Start the MLflow UI with: `mlflow ui` then open http://127.0.0.1:5000
#
# Everything else (model, agent, prompts) stays identical.

#%% [markdown]
# ## Imports

import asyncio
from pathlib import Path

import mlflow
import mlflow.pydantic_ai          # registers the autolog integration

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


#%% [markdown]
# ## MLflow Setup
#
# `autolog()` monkey-patches PydanticAI's agent and model classes so that
# every run is automatically wrapped in an MLflow trace.
# Call this BEFORE creating the agent.

mlflow.pydantic_ai.autolog(log_traces=True)

# Optional: group runs under a named experiment (creates it if it doesn't exist)
mlflow.set_experiment("ocr-math-agent")

#%% [markdown]
# ## Model & Agent (identical to Version 1)

provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

model = OpenAIChatModel("kimi-k2.5:cloud", provider=provider)

agent = Agent(
    model,
    system_prompt=(
        "You are an expert OCR and math assistant. "
        "When you receive an image, do two things:\n"
        "1. Extract ALL visible text exactly as it appears.\n"
        "2. If you find any mathematical expressions or equations, "
        "solve them step by step and show the final result.\n"
        "Keep your response concise and well structured."
    ),
)

#%% [markdown]
# ## Image Analysis Function
#
# `mlflow.start_run()` creates a named run in the UI so you can tell
# individual executions apart.  The autolog patches add child spans
# automatically — you don't need to add any manual instrumentation.

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

    # Every agent.run() inside this block is captured as a MLflow trace
    with mlflow.start_run(run_name="analyze_image"):

        # Log custom metadata about this run
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("model", "kimi-k2.5:cloud")

        result = await agent.run(
            [
                "Please read this image. Extract all text and solve any math you find.",
                BinaryContent(data=image_bytes, media_type=media_type),
            ]
        )

        # Log the output as a metric/artifact so it's searchable in the UI
        mlflow.log_text(result.output, "response.txt")
        

    return result.output


#%% [markdown]
# ## Entry Point

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.png"

    print(f"Analyzing image: {image_path}")
    print("MLflow UI: run `mlflow ui` and open http://127.0.0.1:5000\n")

    response = asyncio.run(analyze_image(image_path))
    print("=" * 50)
    print(response)
