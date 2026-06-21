"""Gemini backend for the Operations Assistant chatbot.

Fails soft on purpose: raises a plain exception on any problem (missing key,
bad model name, network error, rate limit) so the caller can fall back to
the local rule-based assistant. A live operations workspace should never go silent because
of this module.
"""

from __future__ import annotations

from typing import List, Tuple

from src.assistant_context import SYSTEM_INSTRUCTION


def ask_gemini(
    question: str,
    context_json: str,
    history: List[Tuple[str, str]],
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> str:
    """Call the Gemini API. Raises on any failure; caller is responsible for fallback.

    `history` is a list of (role, text) tuples, role being "user" or "model", oldest first.
    """
    if not api_key:
        raise RuntimeError("No Gemini API key configured.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents = []
    for role, text in history[-6:]:
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"Platform data context (JSON):\n{context_json}\n\nQuestion: {question}")],
        )
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=600,
        ),
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text
