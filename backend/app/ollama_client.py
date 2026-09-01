"""
Minimal client for a local Ollama server. No API key, no network egress —
everything runs on http://localhost:11434 (or wherever OLLAMA_HOST points).
"""
from __future__ import annotations
import json
import re
from typing import Optional

import requests

from . import config


class OllamaError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models() -> list[str]:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def chat(model: str, system: str, user: str, max_tokens: int = 1000,
         json_mode: bool = False, temperature: float = 0.1) -> str:
    """Single-turn chat completion against Ollama. Returns the raw text content."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat", json=payload,
            timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise OllamaError(
            f"Could not reach Ollama at {config.OLLAMA_HOST} — is `ollama serve` running "
            f"and has `ollama pull {model}` been run? ({e})"
        )

    if resp.status_code == 404:
        raise OllamaError(
            f"Model '{model}' not found on the Ollama server. Run `ollama pull {model}` first."
        )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def chat_json(model: str, system: str, user: str, max_tokens: int = 1000,
              temperature: float = 0.1) -> dict:
    """Chat completion that asks the model for JSON and parses it, with a
    best-effort fallback if the model wraps it in prose/fences anyway."""
    raw = chat(model, system, user, max_tokens=max_tokens, json_mode=True, temperature=temperature)
    cleaned = strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}
