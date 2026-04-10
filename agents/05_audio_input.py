'''
A PydanticAI agent that records audio from the local microphone,
transcribes and responds to it — powered by Ollama + Gemma 4.
The agent's text reply is also spoken back via the macOS `say` command.
'''
#%% [markdown]
# # Version 5: Audio Input + Output Agent
#
# This script demonstrates how to pass audio data to a PydanticAI agent
# and speak the response back to the user:
#
# - **Model**: Gemma 4 (local edge model) via Ollama (OpenAI-compatible API)
# - **Input**: Raw audio recorded from the laptop microphone
# - **Transport**: `BinaryContent` wraps WAV bytes — same mechanism as images
# - **Output**: Agent's text reply is spoken aloud via the macOS `say` command
# - **Monitoring**: MLflow autolog traces every run
# - **Task**: Listen to the user's voice and respond

#%% [markdown]
# ## Imports

import asyncio
import io
import subprocess
import sys

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import mlflow
import mlflow.pydantic_ai          # registers the autolog integration

#%% [markdown]
# ## MLflow Setup

mlflow.set_experiment("audio-agent")
mlflow.pydantic_ai.autolog(log_traces=True)

#%% [markdown]
# ## Model Setup
#
# `gemma4:e4b` is a local multimodal edge model served through Ollama.
# We point PydanticAI's `OpenAIProvider` at the local Ollama REST API.

provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",           # Ollama ignores the key, but the field is required
)

model = OpenAIChatModel("gemma4:e4b", provider=provider)

#%% [markdown]
# ## Agent Definition

agent = Agent(
    model,
    system_prompt=(
        "You are a helpful voice assistant. "
        "Listen carefully to the audio the user provides and respond "
        "naturally to what they said. Keep your reply concise."
    ),
)

#%% [markdown]
# ## Audio Recording
#
# `sounddevice.rec` captures mono PCM samples at 16 kHz (the standard rate
# for speech models). `scipy.io.wavfile` encodes the numpy array into a WAV
# byte stream that BinaryContent can carry.

def record_audio(duration: int = 5, samplerate: int = 16_000) -> bytes:
    """Record from the default microphone and return WAV-encoded bytes."""
    print(f"Recording for {duration} second(s) … speak now!")
    audio = sd.rec(
        duration * samplerate,
        samplerate=samplerate,
        channels=1,
        dtype="int16",
    )
    sd.wait()                       # block until recording is complete
    print("Recording done.")

    buf = io.BytesIO()
    wavfile.write(buf, samplerate, audio)
    return buf.getvalue()

#%% [markdown]
# ## Audio Output
#
# macOS ships with the `say` command-line tool which converts text to speech
# using the system's built-in voices — no extra dependency required.
# `subprocess.run` blocks until the speech is finished.

def speak(text: str) -> None:
    """Speak text aloud using the macOS `say` command."""
    subprocess.run(["say", text], check=True)

#%% [markdown]
# ## Main Function
#
# `BinaryContent` wraps the raw WAV bytes with the correct MIME type so the
# model receives the audio alongside the text instruction — exactly the same
# pattern used for images in the earlier examples.
# After printing the reply, `speak()` reads it back to the user.

async def main(duration: int = 5) -> None:
    wav_bytes = record_audio(duration)

    result = await agent.run(
        [
            "Please listen to this audio and respond to what you hear.",
            BinaryContent(data=wav_bytes, media_type="audio/wav"),
        ]
    )

    print("\n" + "=" * 50)
    print(result.output)
    speak(result.output)


#%% [markdown]
# ## Entry Point
#
# Optional CLI argument: recording duration in seconds (default 5).
# Example: `uv run 05_audio_input.py 10`

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(main(duration))
