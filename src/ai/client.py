"""Groq API client wrapping Llama 3.3 70B for the AI Analyst tab.

Thin wrapper. Reads GROQ_API_KEY from the environment (loaded via dotenv in
app.py). Caller is responsible for prompt construction — see src/ai/prompts.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from groq import APIError, Groq

MODEL = "llama-3.3-70b-versatile"


@dataclass
class AIResponse:
    text: str
    model: str
    error: str | None = None


class GroqClient:
    """Minimal Groq chat wrapper.

    Raises GroqUnavailable if no API key is configured. Network/API errors
    bubble up so the UI can render a friendly fallback.
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise GroqUnavailable(
                "GROQ_API_KEY is not set. Add it to .env "
                "(register free at https://console.groq.com/keys)."
            )
        self._client = Groq(api_key=key)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> AIResponse:
        try:
            resp = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return AIResponse(text=resp.choices[0].message.content, model=MODEL)
        except APIError as exc:
            return AIResponse(text="", model=MODEL, error=f"Groq API error: {exc}")
        except Exception as exc:  # network, timeout, etc.
            return AIResponse(text="", model=MODEL, error=f"AI unavailable: {exc}")


class GroqUnavailable(RuntimeError):
    """Raised when no API key is configured — UI should degrade gracefully."""
