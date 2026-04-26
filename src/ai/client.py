"""AI client with three-provider fallback chain.

Priority order:
  1. Groq  — Llama 3.3 70B (fastest, free tier)
  2. xAI   — Grok (OpenAI-compatible API, fallback)
  3. Gemini — Gemini 2.0 Flash (Google, last resort)

Reads keys from environment:
  GROQ_API_KEY, XAI_API_KEY, GEMINI_API_KEY

Never hardcode keys — load via .env or deployment secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from groq import APIError, Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

# xAI model candidates — tried in order until one works
_XAI_MODELS = ["grok-2-1212", "grok-beta", "grok-2-vision-1212"]

# Gemini model candidates — tried in order until one works
_GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]


@dataclass
class AIResponse:
    text: str
    model: str
    error: str | None = None


class GroqUnavailable(RuntimeError):
    """Raised when NO provider key is configured — UI degrades gracefully."""


# ── Provider helpers ────────────────────────────────────────────────────────

def _try_groq(messages: list[dict], temperature: float, max_tokens: int) -> AIResponse | None:
    """Returns AIResponse on success, None if key missing, AIResponse(error=) on failure."""
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
        return AIResponse(text="", model=GROQ_MODEL, error=str(exc))
    except Exception as exc:
        return AIResponse(text="", model=GROQ_MODEL, error=str(exc))


def _try_xai(messages: list[dict], temperature: float, max_tokens: int) -> AIResponse | None:
    """xAI Grok via OpenAI-compatible endpoint. Tries multiple model names."""
    key = os.environ.get("XAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
    last_error = ""
    for model_name in _XAI_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return AIResponse(text=resp.choices[0].message.content, model=model_name)
        except Exception as exc:
            last_error = str(exc)
            continue  # try next model name

    return AIResponse(text="", model="xAI", error=last_error)


def _try_gemini(messages: list[dict], temperature: float, max_tokens: int) -> AIResponse | None:
    """Google Gemini. Tries multiple model names until one is available."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        return None

    genai.configure(api_key=key)

    # Build a single prompt string from OpenAI-style messages
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = "\n\n".join(
        f"[{m['role'].upper()}]\n{m['content']}"
        for m in messages if m["role"] != "system"
    )
    prompt = ((system_parts[0] + "\n\n") if system_parts else "") + convo

    last_error = ""
    for model_name in _GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            resp = model.generate_content(prompt)
            return AIResponse(text=resp.text, model=model_name)
        except Exception as exc:
            last_error = str(exc)
            continue  # try next model name

    return AIResponse(text="", model="Gemini", error=last_error)


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
                continue          # key not set — skip silently
            if result.error:
                errors.append(f"{label}: {result.error}")
                continue          # error — try next provider
            return result         # success

        if not errors:
            raise GroqUnavailable(
                "No AI provider keys found. Set at least one of: "
                "GROQ_API_KEY, XAI_API_KEY, GEMINI_API_KEY in your Streamlit secrets."
            )
        raise GroqUnavailable(
            "All AI providers failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )
