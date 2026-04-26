"""AI client with three-provider fallback chain.

Priority order:
  1. Groq  — Llama 3.3 70B (fastest, free tier)
  2. xAI   — Grok 2 (OpenAI-compatible API, fallback)
  3. Gemini — Gemini 1.5 Flash (Google, last resort)

Reads keys from environment:
  GROQ_API_KEY, XAI_API_KEY, GEMINI_API_KEY

Never hardcode keys — load via .env or deployment secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── Groq ───────────────────────────────────────────────────────────────────
from groq import APIError, Groq

GROQ_MODEL = "llama-3.3-70b-versatile"
XAI_MODEL  = "grok-2-latest"
GEMINI_MODEL = "gemini-1.5-flash"


@dataclass
class AIResponse:
    text: str
    model: str
    error: str | None = None


class GroqUnavailable(RuntimeError):
    """Raised when NO provider key is configured — UI degrades gracefully."""


# ── Provider helpers ────────────────────────────────────────────────────────

def _try_groq(messages: list[dict], temperature: float, max_tokens: int) -> AIResponse | None:
    """Returns AIResponse on success, None if key missing, raises on API error."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return AIResponse(text=resp.choices[0].message.content, model=GROQ_MODEL)
    except APIError as exc:
        # Rate limit or auth error — fall through to next provider
        return AIResponse(text="", model=GROQ_MODEL, error=str(exc))
    except Exception as exc:
        return AIResponse(text="", model=GROQ_MODEL, error=str(exc))


def _try_xai(messages: list[dict], temperature: float, max_tokens: int) -> AIResponse | None:
    """xAI Grok via OpenAI-compatible endpoint. Returns None if key missing."""
    key = os.environ.get("XAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=key,
            base_url="https://api.x.ai/v1",
        )
        resp = client.chat.completions.create(
            model=XAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return AIResponse(text=resp.choices[0].message.content, model=XAI_MODEL)
    except ImportError:
        return None  # openai package not installed
    except Exception as exc:
        return AIResponse(text="", model=XAI_MODEL, error=str(exc))


def _try_gemini(messages: list[dict], temperature: float, max_tokens: int) -> AIResponse | None:
    """Google Gemini via google-generativeai. Returns None if key missing."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        # Convert OpenAI-style messages to a single prompt string for Gemini
        prompt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}"
            for m in messages
            if m["role"] != "system"
        )
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        if system_parts:
            prompt = system_parts[0] + "\n\n" + prompt
        resp = model.generate_content(prompt)
        return AIResponse(text=resp.text, model=GEMINI_MODEL)
    except ImportError:
        return None  # google-generativeai not installed
    except Exception as exc:
        return AIResponse(text="", model=GEMINI_MODEL, error=str(exc))


# ── Public client ───────────────────────────────────────────────────────────

class GroqClient:
    """Thin AI client with automatic three-provider fallback.

    Tries Groq → xAI → Gemini in order. Falls through to the next
    provider when a key is missing OR when the API returns an error.
    Raises GroqUnavailable only when every configured provider fails.
    """

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> AIResponse:
        errors: list[str] = []

        for provider_fn, label in [
            (_try_groq,   "Groq"),
            (_try_xai,    "xAI"),
            (_try_gemini, "Gemini"),
        ]:
            result = provider_fn(messages, temperature, max_tokens)
            if result is None:
                # Key not set — skip silently
                continue
            if result.error:
                errors.append(f"{label}: {result.error}")
                # Error but key was present — try next provider
                continue
            # Success
            return result

        # All providers failed or no keys configured
        if not errors:
            raise GroqUnavailable(
                "No AI provider keys found. Set at least one of: "
                "GROQ_API_KEY, XAI_API_KEY, GEMINI_API_KEY in your .env file."
            )
        raise GroqUnavailable(
            "All AI providers failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )
