"""Compatibility exports; all LLM transport lives in llm_client.py."""
from .llm_client import LLMError, chat, chat_json, is_available, list_models, strip_json_fences
OllamaError = LLMError
