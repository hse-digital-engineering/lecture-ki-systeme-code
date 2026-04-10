# Agents

The examples make use of ollama cloud models. Make sure that ollama is installed on the host machine and has internet connection.

The example model use is `kimi-k2.5:cloud`. It also works with `gemma4:e4b` as local edge model.

## Examples

- use `uv run mlfow ui` to open the agent monitoring
- use `uv run <agent.py> Image.jpg` to launch the agents 1-3.
- use `uv run 04_skilled-agent.py` to launch the agents 4 it also offers a webfrontend
- use `uv run 05_audio_input.py` (optional: pass seconds as argument, default 5) to launch the audio agent

This folder contains 5 versions of agents.

1. [Basic Agent without monitoring](01_agent.py)
2. [Agent with mlflow monitoring](02_agent_mlflow.py)
3. [Agent with mlflow monitoring and mcp server](03_agent_mcp.py)

all agents analyse an image and calculate the math equation on it.

1. [Agent with skill](04_skilled-agent.py)
2. [Agent with audio input from the microphone](05_audio_input.py)

The agent skill example relates to the example in the agent skills package https://github.com/DougTrajano/pydantic-ai-skills/tree/main implements the skill format https://agentskills.io/what-are-skills

## Skill template

The folder [skills/my-skill](skills/my-skill) offers the basic skill template based on the pydantic-ai-skills

## Note 

As of April 2026 mlflow requires a pydantic-ai version < 1.68

