
"""Basic example demonstrating skill integration with Pydantic AI.

This example shows how to create an agent with skills and use them
for research tasks.
"""

from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


import mlflow
import mlflow.pydantic_ai          # registers the autolog integration

from pydantic_ai_skills import SkillsToolset

load_dotenv()

mlflow.set_experiment("skilled-agent")
mlflow.pydantic_ai.autolog(log_traces=True)

# Get the skills directory (examples/skills)
skills_dir = Path(__file__).parent / 'skills'

# Initialize Skills Toolset
skills_toolset = SkillsToolset(directories=[skills_dir])

provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

model = OpenAIChatModel("kimi-k2.5:cloud", provider=provider)

# Create agent with skills
agent = Agent(
    model=model,
    instructions='You are a helpful research assistant.',
    toolsets=[skills_toolset],
)


app = agent.to_web()


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7932)