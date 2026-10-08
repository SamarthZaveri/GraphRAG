"""Chat-completion client for hosted APIs and optional local Ollama."""
import json
import re
import time
import requests
from . import config


class LLMError(RuntimeError):
    pass


def _headers():
    return {"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}


def is_available():
    if config.LLM_BACKEND == "api" and not config.API_KEY:
        return False
    try:
        url = config.API_BASE_URL + "/models" if config.LLM_BACKEND == "api" else config.OLLAMA_HOST + "/api/tags"
        response = requests.get(url, headers=_headers() if config.LLM_BACKEND == "api" else {}, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_models():
    try:
        url = config.API_BASE_URL + "/models" if config.LLM_BACKEND == "api" else config.OLLAMA_HOST + "/api/tags"
        response = requests.get(url, headers=_headers() if config.LLM_BACKEND == "api" else {}, timeout=10)
        response.raise_for_status()
        data = response.json()
        return ([m["id"] for m in data.get("data", [])] if config.LLM_BACKEND == "api"
                else [m["name"] for m in data.get("models", [])])
    except (requests.RequestException, ValueError, KeyError):
        return []


def chat(model, system, user, max_tokens=1000, json_mode=False, temperature=0.1):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if config.LLM_BACKEND == "api":
        if not config.API_KEY:
            raise LLMError("Set HF_TOKEN or LEDGER_API_KEY before using the hosted API.")
        url = config.API_BASE_URL + "/chat/completions"
        headers = _headers()
        payload = {"model": model, "messages": messages, "stream": False,
                   "max_tokens": max_tokens, "temperature": temperature}
        # Some hosted providers do not support response_format; prompts still require JSON.
        if json_mode and config.API_JSON_MODE:
            payload["response_format"] = {"type": "json_object"}
    else:
        url = config.OLLAMA_HOST + "/api/chat"
        headers = {}
        payload = {"model": model, "messages": messages, "stream": False,
                   "keep_alive": config.OLLAMA_KEEP_ALIVE,
                   "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": config.OLLAMA_NUM_CTX}}
        if json_mode:
            payload["format"] = "json"
    for attempt in range(max(1, config.LLM_RETRIES)):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=config.OLLAMA_REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt + 1 >= max(1, config.LLM_RETRIES):
                raise LLMError(f"LLM request failed for {model}; check endpoint/network and timeout.") from None
            time.sleep(min(2 ** attempt, 8))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 < max(1, config.LLM_RETRIES):
                time.sleep(min(2 ** attempt, 8))
                continue
        if not response.ok:
            raise LLMError(f"LLM HTTP {response.status_code} for {model}; check model availability, credentials, credits, and provider limits.")
        try:
            data = response.json()
            if config.LLM_BACKEND == "api":
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise LLMError(f"Output truncated for {model}; increase max_tokens or shorten input.")
                content = choice["message"].get("content")
            else:
                if data.get("done_reason") == "length":
                    raise LLMError(f"Output truncated for {model}; increase max_tokens or shorten input.")
                content = data.get("message", {}).get("content")
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError(f"Invalid chat response from {model}.") from None
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Empty chat response from {model}.")
        return content


def strip_json_fences(text):
    return re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", text.strip())).strip()


def chat_json(model, system, user, max_tokens=1000, temperature=0.1):
    raw = strip_json_fences(chat(model, system, user, max_tokens, True, temperature))
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except ValueError:
        # Decode an embedded object without a greedy regex swallowing multiple objects.
        for position, char in enumerate(raw):
            if char == "{":
                try:
                    data, _ = json.JSONDecoder().raw_decode(raw[position:])
                    if isinstance(data, dict):
                        return data
                except ValueError:
                    continue
        return {}
