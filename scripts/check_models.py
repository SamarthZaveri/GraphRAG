"""Check model configuration, network access and optional hosted completions."""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import config, llm_client


def check_network():
    """Probe the configured endpoint without sending an API credential."""
    url = config.API_BASE_URL + "/models" if config.LLM_BACKEND == "api" else config.OLLAMA_HOST + "/api/tags"
    print("Network host:", urlsplit(url).hostname, flush=True)
    print("Proxy configured:", bool(requests.utils.get_environ_proxies(url)), flush=True)
    try:
        response = requests.get(url, timeout=(10, 20), allow_redirects=False)
    except requests.RequestException as error:
        raise llm_client.LLMError(llm_client.transport_error(error, url)) from None
    print(f"Network probe: HTTP {response.status_code} (no credential sent)", flush=True)
    if response.status_code in (401, 403):
        print("  HTTPS works; this route requires authentication or access. The smoke test checks your key.")
    elif not response.ok:
        raise llm_client.LLMError(f"Network probe reached the server but returned HTTP {response.status_code}; check the endpoint/provider.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-test", action="store_true", help="Probe the endpoint without sending your API key")
    parser.add_argument("--smoke-test", action="store_true", help="Make one small completion per unique configured model; uses provider quota")
    parser.add_argument("--timeout", type=int, default=120, help="Smoke-test response timeout in seconds (default: 120)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    models = sorted({config.EXTRACTION_MODEL, config.ANSWER_MODEL, config.QUESTION_MODEL, config.JUDGE_MODEL})
    print(f"Backend: {config.LLM_BACKEND}; default profile: {config.MODEL_PROFILE} (role overrides take precedence)", flush=True)
    print("API credential configured:", bool(config.API_KEY), flush=True)
    try:
        if args.network_test:
            check_network()
        # Diagnostics fail promptly instead of retrying a long completion three times.
        if args.smoke_test:
            config.OLLAMA_REQUEST_TIMEOUT = args.timeout
            config.LLM_RETRIES = 1
            config.RATE_LIMIT_ATTEMPTS = 1
        for model in models:
            print("Model:", model, flush=True)
            if config.LLM_BACKEND == "api" and "router.huggingface.co" in config.API_BASE_URL:
                response = requests.get("https://huggingface.co/api/models/" + model.split(":")[0],
                                        params={"expand": "inferenceProviderMapping"}, timeout=(10, 30))
                response.raise_for_status()
                mapping = response.json().get("inferenceProviderMapping", {})
                providers = [name for name, details in mapping.items() if details.get("status") == "live"]
                print("  Live providers:", ", ".join(providers) or "none; use a dedicated endpoint or another model")
            if args.smoke_test:
                result = llm_client.chat_json(model, 'Return only JSON with an "ok" boolean field.',
                                              'Reply with {"ok": true} only.', max_tokens=256, temperature=0)
                if result.get("ok") is not True:
                    raise llm_client.LLMError(f"JSON smoke test failed for {model}; received no valid ok=true object")
                print("  JSON completion succeeded")
    except llm_client.LLMError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except requests.RequestException as error:
        print("ERROR:", llm_client.transport_error(error, "https://huggingface.co"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
