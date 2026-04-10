'''
A minimal PydanticAI agent that reads an image, extracts text,
and solves any math it finds — powered by Ollama + Kimi K2.5.
'''
#%% [markdown]
# # Version 1: The Agent
#
# This script shows the simplest possible PydanticAI agent setup:
#
# - **Model**: Kimi K2.5 via Ollama (OpenAI-compatible API)
# - **System prompt**: defines the agent's role
# - **User prompt**: image + task description
# - **Task**: extract text from an image and solve any math

#%% [markdown]
# ## Imports

import asyncio
from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


#%% [markdown]
# ## Model Setup
#
# Ollama exposes an OpenAI-compatible REST API at `localhost:11434`.
# We point PydanticAI's `OpenAIProvider` there so no real API key is needed.

provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",           # Ollama ignores the key, but the field is required
)

model = OpenAIChatModel("kimi-k2.5:cloud", provider=provider)

#%% [markdown]
# ## Agent Definition
#
# The system prompt sets up the agent's persona and task.
# The agent will be called with a user prompt that includes the image.

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
# `BinaryContent` wraps raw image bytes so the model can see the image.
# The user prompt and the image are passed together as a list.

async def analyze_image(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()

    # Detect media type from file extension
    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    result = await agent.run(
        [
            "Please read this image. Extract all text and solve any math you find.",
            BinaryContent(data=image_bytes, media_type=media_type),
        ]
    )
    return result.output


#%% [markdown]
# ## Entry Point

if __name__ == "__main__":
    import sys

    # Accept image path as CLI argument, fall back to "sample.png"
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.png"

    print(f"Analyzing image: {image_path}\n")
    response = asyncio.run(analyze_image(image_path))
    print("=" * 50)
    print(response)
