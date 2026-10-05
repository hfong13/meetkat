"""
The only file that talks to an LLM API (Anthropic's Claude or Google's Gemini).

It uses Python's standard library (urllib) instead of an SDK: no extra
dependency, and every byte we send is visible right here.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import config


class LLMError(Exception):
    """One call failed. The caller can skip this item and carry on."""


class FatalLLMError(LLMError):
    """Retrying is pointless (no key, key rejected). Stop the whole run."""


def provider():
    """Which API to use: Claude if ANTHROPIC_API_KEY is set, otherwise Gemini
    if GEMINI_API_KEY is set. Force one with MEETKAT_PROVIDER=anthropic|gemini."""
    forced = os.environ.get("MEETKAT_PROVIDER", "").lower()
    if forced in config.PROVIDERS:
        return forced
    for name, settings in config.PROVIDERS.items():
        if os.environ.get(settings["key_env_var"]):
            return name
    names = " or ".join(p["key_env_var"] for p in config.PROVIDERS.values())
    raise FatalLLMError(f"Set the {names} environment variable first.")


def model_name():
    """e.g. 'gemini/gemini-3.5-flash-lite'. Saved in outputs and used in cache keys,
    so answers from different models never get mixed up."""
    name = provider()
    return f"{name}/{config.PROVIDERS[name]['model']}"


def call(system, messages, json_schema=None, max_tokens=400):
    """Send one request to the chosen LLM API and return the reply text.

    messages: [{"role": "user" | "assistant", "content": "..."}].
    If json_schema is given we ask for "structured output": the API constrains
    the reply so it matches the schema.
    """
    name = provider()
    settings = config.PROVIDERS[name]
    api_key = os.environ.get(settings["key_env_var"])
    if not api_key:
        raise FatalLLMError(f"Set the {settings['key_env_var']} environment variable first.")
    if name == "gemini":
        return _call_gemini(settings, api_key, system, messages, json_schema, max_tokens)
    return _call_anthropic(settings, api_key, system, messages, json_schema, max_tokens)


def _call_anthropic(settings, api_key, system, messages, json_schema, max_tokens):
    payload = {
        "model": settings["model"],
        "max_tokens": max_tokens,
        "temperature": 0,        # least random output: same input -> (almost) same answer
        "system": system,
        "messages": messages,
    }
    if json_schema:
        payload["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
    request = urllib.request.Request(
        settings["url"], data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    reply = _send_with_retries(request)

    # A reply cut off by max_tokens, or a refusal, may not match the schema.
    if reply.get("stop_reason") in ("max_tokens", "refusal"):
        raise LLMError(f"reply stopped early ({reply['stop_reason']})")
    text = "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text")
    if not text.strip():
        raise LLMError("empty reply")
    return text


def gemini_schema(schema):
    """Gemini's responseSchema uses an older schema dialect: types are written
    in capitals (OBJECT, STRING...) and 'additionalProperties' isn't allowed.
    Our own validator still enforces the full schema afterwards."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue
            out[key] = value.upper() if key == "type" and isinstance(value, str) else gemini_schema(value)
        return out
    if isinstance(schema, list):
        return [gemini_schema(v) for v in schema]
    return schema


def gemini_payload(system, messages, json_schema, max_tokens):
    """Turn our Claude-style messages into Gemini's request format."""
    config_block = {"temperature": 0, "maxOutputTokens": max_tokens}
    if json_schema:
        config_block["responseMimeType"] = "application/json"
        config_block["responseSchema"] = gemini_schema(json_schema)
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]} for m in messages],
        "generationConfig": config_block,
    }


def gemini_text(reply):
    """Pull the answer text out of a Gemini reply, or raise LLMError."""
    candidates = reply.get("candidates") or []
    if not candidates:
        blocked = reply.get("promptFeedback", {}).get("blockReason")
        raise LLMError(f"no answer (blocked: {blocked})" if blocked else "empty reply")
    candidate = candidates[0]
    finish = candidate.get("finishReason", "STOP")
    if finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
        raise LLMError(f"reply stopped early ({finish})")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))   # skip "thinking" parts
    if not text.strip():
        raise LLMError("empty reply")
    return text


def _call_gemini(settings, api_key, system, messages, json_schema, max_tokens):
    url = settings["url"].format(model=settings["model"])
    # Some newer Gemini models "think" before answering and that counts towards
    # the token limit, so we allow more room than the answer itself needs.
    payload = gemini_payload(system, messages, json_schema, max(max_tokens, 2048))
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"x-goog-api-key": api_key, "content-type": "application/json"})
    return gemini_text(_send_with_retries(request))


def _send_with_retries(request):
    """Retry only errors that might go away by themselves: rate limits (429),
    timeouts, server/overload errors (5xx) and network drops. Wait 2s, then 4s
    ("exponential backoff") so we don't hammer a struggling server."""
    for attempt in range(1, config.MAX_API_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        # HTTPError must come before URLError: it's a subclass of it.
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            if err.code in (401, 403) or "API_KEY_INVALID" in detail:   # Gemini sends 400 for a bad key
                raise FatalLLMError(f"API key rejected (HTTP {err.code}): {detail}")
            retryable = err.code in (408, 429) or err.code >= 500
            if not retryable or attempt == config.MAX_API_ATTEMPTS:
                raise LLMError(f"HTTP {err.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ValueError) as err:
            if attempt == config.MAX_API_ATTEMPTS:
                raise LLMError(f"network or response error: {err}")
        time.sleep(2 ** attempt)


def ask_validated(system, user_text, validate, json_schema=None, call_fn=None):
    """Ask once. If validate() finds problems, show the model its own reply,
    say exactly what was wrong, and ask ONE more time.

    validate(text) must return (value, errors); errors == [] means success.
    call_fn lets tests swap in a fake API.
    """
    call_fn = call_fn or call
    messages = [{"role": "user", "content": user_text}]
    text = call_fn(system, messages, json_schema)
    value, errors = validate(text)
    if not errors:
        return value

    messages += [
        {"role": "assistant", "content": text},
        {"role": "user", "content": "Your reply was invalid: " + "; ".join(errors)
                                    + ". Reply again, fixing only these problems."},
    ]
    text = call_fn(system, messages, json_schema)
    value, errors = validate(text)
    if errors:
        raise LLMError("still invalid after one retry: " + "; ".join(errors))
    return value


# ------------------------------------------------------------------ cache
def cache_key(*parts):
    """A fingerprint of everything that affects the answer (model, prompt,
    schema, input). Change any of them and the key changes, so we re-ask."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def load_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache, path):
    """Write to a temporary file, then rename it over the old one. The rename
    is atomic, so a crash halfway through can't leave a corrupted cache."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, path)
