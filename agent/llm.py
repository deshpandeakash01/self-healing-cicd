"""
Creates the LLM the agent uses (Google Gemini through LangChain).

Kept in its own small file so the rest of the agent never mentions a vendor:
swapping the model, or the provider, means changing only this file.
"""

from __future__ import annotations

import os

from . import config


def make_llm(model: str | None = None):
    # Imported here so that classifier_only mode and the tests do not need
    # the Gemini package installed.
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError(
            "No Gemini API key found. Create one in Google AI Studio and set "
            "it as the GOOGLE_API_KEY environment variable.")
    extra = {}
    if config.GEMINI_THINKING_LEVEL:
        extra["thinking_config"] = {"thinking_level": config.GEMINI_THINKING_LEVEL}
    return ChatGoogleGenerativeAI(
        model=model or config.GEMINI_MODEL,
        temperature=0,        # same input should give the same diagnosis
        max_retries=2,
        **extra,
    )


def model_name(llm) -> str | None:
    """
    The model an LLM object talks to, e.g. "gemini-4-hydrogen".

    Read from the object itself, not from config, so the audit log and the
    evaluation record what was really used. None for an object that does not
    say (the stand-in LLM in the tests).
    """
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    if not isinstance(name, str) or not name:
        return None
    return name.removeprefix("models/")


def thinking_level(llm) -> str | None:
    """The thinking level an LLM object was created with, None if not set."""
    cfg = getattr(llm, "thinking_config", None)
    if cfg is None:
        return None
    level = (cfg.get("thinking_level") if isinstance(cfg, dict)
             else getattr(cfg, "thinking_level", None))
    return str(level).lower() if level else None
