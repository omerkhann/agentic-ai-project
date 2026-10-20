"""LLM provider interface. Plain HTTP (httpx), no vendor SDKs, all free backends.

  LLM_PROVIDER=ollama      local Ollama server (default model qwen2.5-coder:7b)
  LLM_PROVIDER=gemini      Google AI Studio free tier, OpenAI-compatible endpoint
  LLM_PROVIDER=groq        Groq free tier
  LLM_PROVIDER=openrouter  OpenRouter free models

Settings come from .env (see .env.example).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx
from dotenv import load_dotenv

from src import config

load_dotenv(config.ROOT / ".env")

Messages = list[dict[str, str]]

_OPENAI_COMPAT = {
    # name: (base_url, api-key env var, model env var, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "GEMINI_MODEL",
               "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.1-8b-instant"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
                   "qwen/qwen-2.5-coder-32b-instruct:free"),
}


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, messages: Messages, json_mode: bool = False) -> str:
        """Return the assistant's text reply."""


class OllamaProvider(LLMProvider):
    def __init__(self, host: str | None = None, model: str | None = None, num_ctx: int = 8192):
        self.name = "ollama"
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.num_ctx = num_ctx

    def complete(self, messages: Messages, json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        if json_mode:
            body["format"] = "json"
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str):
        base, key_var, model_var, default_model = _OPENAI_COMPAT[name]
        key = os.getenv(key_var)
        if not key:
            raise RuntimeError(f"LLM_PROVIDER={name} needs {key_var} in .env")
        self.name, self.base, self.key = name, base, key
        self.model = os.getenv(model_var) or default_model

    def complete(self, messages: Messages, json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = httpx.post(f"{self.base}/chat/completions", json=body, timeout=120,
                       headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if name == "ollama":
        return OllamaProvider()
    if name in _OPENAI_COMPAT:
        return OpenAICompatProvider(name)
    raise ValueError(f"unknown LLM_PROVIDER {name!r}; use ollama | {' | '.join(_OPENAI_COMPAT)}")
