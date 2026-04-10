# Agents

- use `uv run mlfow ui` to open the agent monitoring
- use `uv run <agent.py> Image.jpg` to launch the agents 1-3.
- use `uv run 04_skilled-agent.py` to launch the agents 4 it also offers a webfrontend

This folder contains 4 versions of agents.

1. [Basic Agent without monitoring](01_agent.py)
2. [Agent with mlflow monitoring](02_agent_mlflow.py)
3. [Agent with mlflow monitoring and mcp server](03_agent_mcp.py)

all agents analyse an image and calculate the math equation on it.

1. [Agent with skill](04_skilled-agent.py)

The agent skill example relates to the example in the agent skills package https://github.com/DougTrajano/pydantic-ai-skills/tree/main implements the skill format https://agentskills.io/what-are-skills

## Skill template

The folder [skills/my-skill](skills/my-skill) offers the basic skill template based on the pydantic-ai-skills

## Note 

As of April 2026 mlflow requires a pydantic-ai version < 1.68

