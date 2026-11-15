"""Check which OpenRouter models used by the app are working.

    python check_openrouter_models.py           # catalogue check (no API key needed)
    python check_openrouter_models.py --live    # also send a tiny test prompt to each model

The catalogue check confirms every model ID still exists on OpenRouter and
has at least one provider endpoint. `--live` additionally calls each model
with the OpenRouter key you configured (config.json / OPENROUTER_API_KEY)
and reports the exact error for any model that fails.
"""

import argparse
import json
import sys
import urllib.request

from modules.config import (OPENROUTER_LLM_MODELS, OPENROUTER_OCR_MODELS,
                            get_openrouter_api_key, get_openrouter_base_url,
                            get_openrouter_llm_model, get_openrouter_ocr_model)

PUBLIC_API = "https://openrouter.ai/api/v1"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def all_models() -> list:
    """Every model the app can use: UI lists plus the config.json defaults."""
    seen, out = set(), []
    for m in [get_openrouter_llm_model(), get_openrouter_ocr_model(),
              *OPENROUTER_LLM_MODELS, *OPENROUTER_OCR_MODELS]:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def catalogue_check(models: list) -> dict:
    catalogue = {m["id"]: m for m in _get_json(f"{PUBLIC_API}/models")["data"]}
    results = {}
    for model in models:
        if model not in catalogue:
            results[model] = "not a valid OpenRouter model ID"
            continue
        try:
            eps = _get_json(f"{PUBLIC_API}/models/{model}/endpoints")["data"]["endpoints"]
        except Exception as exc:  # noqa: BLE001
            results[model] = f"could not read endpoints: {exc}"
            continue
        results[model] = None if eps else "no provider endpoints available"
    return results


def live_check(models: list) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=get_openrouter_api_key(),
                    base_url=get_openrouter_base_url())
    results = {}
    for model in models:
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=256, temperature=0,
                messages=[{"role": "user", "content": "Reply with the word OK."}])
            msg = r.choices[0].message
            ok = (msg.content or getattr(msg, "reasoning", None) or "").strip()
            results[model] = None if ok else "empty response"
        except Exception as exc:  # noqa: BLE001
            results[model] = str(exc)[:300]
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="send a test prompt to each model (uses your OpenRouter key)")
    args = ap.parse_args()

    models = all_models()
    print(f"Checking {len(models)} models against the OpenRouter catalogue...")
    results = catalogue_check(models)

    if args.live:
        if not get_openrouter_api_key():
            print("--live skipped: no OpenRouter API key configured "
                  "(config.json api_keys.openrouter or OPENROUTER_API_KEY).")
        else:
            print("Sending a test prompt to each model...")
            for model, err in live_check([m for m, e in results.items() if e is None]).items():
                results[model] = err

    width = max(len(m) for m in models)
    for model in models:
        err = results[model]
        print(f"  {'OK  ' if err is None else 'FAIL'}  {model:<{width}}  {err or ''}")
    failed = [m for m, e in results.items() if e]
    print(f"\n{len(models) - len(failed)} working, {len(failed)} failing.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
