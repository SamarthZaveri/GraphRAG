"""Chat-completion client for hosted APIs and optional local Ollama."""
import json
import re
import time
import threading
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
import requests
from . import config


class LLMError(RuntimeError):
    pass


class OutputTruncatedError(LLMError):
    """The provider stopped generation at the output-token limit."""
    pass


def transport_error(error, url):
    """Describe transport failures without exposing URLs, headers or credentials."""
    host = urlsplit(url).hostname or "configured endpoint"
    if isinstance(error, requests.exceptions.ProxyError):
        reason = "Proxy connection failed; check HTTPS_PROXY/HTTP_PROXY and your proxy access."
    elif isinstance(error, requests.exceptions.SSLError):
        reason = "TLS handshake or certificate validation failed; check your trusted CA certificates and HTTPS inspection proxy."
    elif isinstance(error, requests.exceptions.ConnectTimeout):
        reason = "Connection timed out; check firewall, VPN and endpoint connectivity."
    elif isinstance(error, requests.exceptions.ReadTimeout):
        reason = "Response timed out; the provider may be busy. Retry later or increase LEDGER_OLLAMA_TIMEOUT."
    elif isinstance(error, requests.exceptions.Timeout):
        reason = "Request timed out; check connectivity or retry later."
    elif isinstance(error, requests.exceptions.ConnectionError):
        # Inspect locally, but never print exception text: proxy URLs may contain passwords.
        detail = str(error).lower()
        if any(marker in detail for marker in ("getaddrinfo", "nameresolution", "name resolution", "nodename nor servname")):
            reason = "DNS lookup failed; check your DNS, VPN and internet connection."
        else:
            reason = "Connection failed or was closed; check firewall/VPN/proxy connectivity and retry."
    else:
        reason = "Request failed; check endpoint and network configuration."
    return f"{host}: {type(error).__name__}. {reason}"


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


def rate_limit_delay(response, attempt):
    """Honor server retry guidance, including HTTP dates; otherwise back off."""
    value = response.headers.get("Retry-After")
    if isinstance(value, str):
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(15 * 2 ** attempt, 120)


_request_lock = threading.Lock()
_next_request_at = 0.0


def pace_request():
    """Limit request starts within this process, including extraction threads."""
    global _next_request_at
    if config.LLM_BACKEND != "api" or config.API_MIN_INTERVAL <= 0:
        return
    with _request_lock:
        now = time.monotonic()
        delay = max(0, _next_request_at - now)
        if delay:
            time.sleep(delay)
        _next_request_at = time.monotonic() + config.API_MIN_INTERVAL


def chat(model, system, user, max_tokens=1000, json_mode=False, temperature=0.1):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if config.LLM_BACKEND == "api":
        if not config.API_KEY:
            raise LLMError("Set HF_TOKEN or LEDGER_API_KEY before using the hosted API.")
        url = config.API_BASE_URL + "/chat/completions"
        headers = _headers()
        payload = {"model": model, "messages": messages, "stream": False,
                   "max_tokens": max_tokens, "temperature": temperature}
        if "integrate.api.nvidia.com" in config.API_BASE_URL and model.startswith("nvidia/nemotron-3"):
            payload["chat_template_kwargs"] = {"enable_thinking": config.NVIDIA_THINKING}
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
    for attempt in range(max(1, config.LLM_RETRIES, config.RATE_LIMIT_ATTEMPTS)):
        try:
            pace_request()
            response = requests.post(url, headers=headers, json=payload, timeout=(10, config.OLLAMA_REQUEST_TIMEOUT))
        except requests.RequestException as error:
            if attempt + 1 >= max(1, config.LLM_RETRIES) or isinstance(error, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
                raise LLMError(f"LLM request failed for {model}. {transport_error(error, url)}") from None
            time.sleep(min(2 ** attempt, 8))
            continue
        if response.status_code == 429:
            delay = rate_limit_delay(response, attempt)
            if attempt + 1 >= config.RATE_LIMIT_ATTEMPTS or delay > 300:
                raise LLMError(f"LLM HTTP 429 for {model}; provider rate limit or quota persists. "
                               f"Retry later (server/backoff delay: {delay:.0f}s), or check your account quota.")
            print(f"  [LLM] HTTP 429 for {model}; waiting {delay:.0f}s before attempt "
                  f"{attempt+2}/{config.RATE_LIMIT_ATTEMPTS}", flush=True)
            time.sleep(delay)
            continue
        if response.status_code >= 500:
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
                    raise OutputTruncatedError(f"Output truncated for {model} at max_tokens={max_tokens}; increase max_tokens or shorten input.")
                content = choice["message"].get("content")
            else:
                if data.get("done_reason") == "length":
                    raise OutputTruncatedError(f"Output truncated for {model} at max_tokens={max_tokens}; increase max_tokens or shorten input.")
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
