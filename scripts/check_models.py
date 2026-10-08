"""Check configured hosted models; optionally make tiny paid API smoke calls."""
import argparse
import sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import config, llm_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Make one small completion per unique configured model; uses API credits")
    args = parser.parse_args()
    models = sorted({config.EXTRACTION_MODEL, config.ANSWER_MODEL, config.QUESTION_MODEL, config.JUDGE_MODEL})
    print(f"Backend: {config.LLM_BACKEND}; profile: {config.MODEL_PROFILE}")
    print("API credential configured:", bool(config.API_KEY))
    for model in models:
        print("Model:", model)
        if config.LLM_BACKEND == "api" and "router.huggingface.co" in config.API_BASE_URL:
            response = requests.get("https://huggingface.co/api/models/" + model.split(":")[0],
                                    params={"expand": "inferenceProviderMapping"}, timeout=30)
            response.raise_for_status()
            mapping = response.json().get("inferenceProviderMapping", {})
            providers = [name for name, details in mapping.items() if details.get("status") == "live"]
            print("  Live providers:", ", ".join(providers) or "none; use a dedicated endpoint or another model")
        if args.smoke_test:
            llm_client.chat(model, "Follow the instruction precisely.", "Reply with OK only.", max_tokens=32, temperature=0)
            print("  Completion succeeded")


if __name__ == "__main__":
    main()
