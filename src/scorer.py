"""
scorer.py
---------
Uses an LLM (local Ollama llama3.1 by default, Groq's openai/gpt-oss-20b
as an automatic cloud fallback) to score a batch of retrieved facets
against a conversation. Never scores more than BATCH_SIZE facets per LLM
call, and never scores medical/biological facets under any circumstances
(defense in depth -- even if one slips past retrieval, we force abstention
here rather than trust the model's judgment on clinical content).

WHY BATCHES OF 10: llama3.1 running locally on a 6GB-VRAM laptop GPU is
noticeably less reliable at returning clean, complete JSON as the requested
output grows. Asking for 25+ facets' worth of structured output in one call
increases truncation/malformed-JSON risk and makes partial-failure recovery
harder (one bad facet shouldn't sink 24 good ones). 10 is small enough for
the model to stay focused and consistent, and large enough to keep the
number of round trips (and prompt-preamble overhead) reasonable. See
docs/DECISIONS.md for more detail.

WHY A HYBRID OLLAMA/GROQ BACKEND: Ollama requires a local GPU-capable
machine, which means the project only runs on hardware like the one it was
built on. Groq's hosted API needs nothing but an API key, so it's a
natural fallback for running this on any machine, in CI, or when the
local Ollama server/GPU isn't available. The specific Groq model
(GROQ_MODEL_NAME, default "openai/gpt-oss-20b") has already changed once --
"llama-3.1-8b-instant" was the original default until it became
enterprise-only and started 404ing on developer accounts, which is
exactly why GROQ_MODEL_NAME is an overridable env var rather than only a
hardcoded constant. See docs/DECISIONS.md #5 for the full reasoning and
trade-offs (in particular: the fallback is silent, so a mid-session
Ollama crash can switch backends without an explicit user action).
"""

import json
import os
import re
from typing import Any

try:
    import ollama
except ImportError as e:
    raise ImportError("ollama package not installed. Run: pip install -r requirements.txt") from e

# Overridable via env var: "localhost" only resolves to the Ollama server
# when this process runs on the same machine as Ollama. Inside a Docker
# container, "localhost" means the container itself -- reaching a host-run
# Ollama needs a different address (e.g. http://host.docker.internal:11434
# on Docker Desktop, or the docker-compose extra_hosts mapping on Linux).
# See docker-compose.yml / docs/README.md "Docker" section.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "llama3.1"
# Overridable via env var on purpose: Groq deprecates/decommissions model
# IDs over time (e.g. "llama3-8b-8192" was shut down 2025-08-30), and access
# tiers change too -- "llama-3.1-8b-instant" was the default here until it
# became enterprise-only and started 404ing on developer accounts. When
# this happens again, set GROQ_MODEL_NAME to whatever's current/accessible
# at https://console.groq.com/docs/models instead of editing this file --
# `python main.py --test-groq` will tell you clearly whether the active
# model name actually works.
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
BATCH_SIZE = 10

VALID_STATUSES = {"scored", "insufficient_evidence", "not_observable"}
VALID_CONFIDENCE = {"high", "medium", "low"}

_client = ollama.Client(host=OLLAMA_HOST)

# Cached result of detect_backend() -- "ollama" | "groq" -- so we don't
# re-probe Ollama's /api/tags on every one of the ~4 batches per
# conversation. None means "not checked yet". Reset with
# detect_backend(force_refresh=True) (used by app.py after the user enters
# a Groq key, and by tests).
_active_backend: str | None = None
_groq_client = None


def _check_ollama_available() -> bool:
    """
    Checks that Ollama is reachable at OLLAMA_HOST (hits /api/tags, same
    endpoint the ollama Python client's .list() call uses under the hood)
    AND that llama3.1 specifically is pulled -- Ollama being *up* isn't
    enough if the model itself isn't available yet.
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models_response = client.list()
        models = models_response.get("models", []) if isinstance(models_response, dict) else models_response.models
        model_names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                model_names.append(name)
        return any("llama3.1" in name for name in model_names)
    except Exception:
        return False


def detect_backend(force_refresh: bool = False) -> str:
    """
    Determine which LLM backend to use, preferring local Ollama:

      1. Ollama reachable + llama3.1 pulled  -> "ollama"
      2. Ollama unavailable, GROQ_API_KEY set -> "groq" (cloud fallback)
      3. Neither                              -> raise RuntimeError with an
                                                  actionable message

    The result is cached at module level after the first successful check
    so repeated calls within one conversation's batches don't re-probe
    Ollama every time. Pass force_refresh=True to bypass the cache (e.g.
    right after the user sets GROQ_API_KEY in app.py, or in tests).
    """
    global _active_backend
    if _active_backend is not None and not force_refresh:
        return _active_backend

    if _check_ollama_available():
        _active_backend = "ollama"
        return _active_backend

    if os.environ.get("GROQ_API_KEY"):
        _active_backend = "groq"
        return _active_backend

    raise RuntimeError(
        "No LLM backend available. Either start Ollama (and run "
        "`ollama pull llama3.1`) so it's reachable at http://localhost:11434, "
        "or set the GROQ_API_KEY environment variable to use the Groq cloud "
        "fallback instead. See docs/README.md 'LLM Backend Options'."
    )


def _get_groq_client():
    """Lazily construct and cache the Groq client so we don't require the
    `groq` package or a valid key unless the Groq backend is actually used."""
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("groq package not installed. Run: pip install -r requirements.txt") from e
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set -- cannot use the Groq backend.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _call_llm(prompt: str) -> str:
    """
    Send one chat-completion request to whichever backend detect_backend()
    resolves to, and return the raw text content. This is the single point
    where score_facet_batch() talks to "the LLM" -- it doesn't need to know
    or care which backend actually served the request.
    """
    backend = detect_backend()
    if backend == "ollama":
        response = _client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},  # low temperature: we want consistent, conservative scoring
        )
        return response["message"]["content"]

    # backend == "groq"
    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
    except Exception as e:
        # A 404 / "model not found" here almost always means GROQ_MODEL_NAME
        # has been deprecated/decommissioned on Groq's side since this was
        # last verified -- not a typo in our code. Re-raise with an
        # actionable hint rather than a bare SDK error; score_facet_batch's
        # caller still catches this the same way either way.
        message = str(e)
        if "404" in message or "does not exist" in message.lower() or "not_found" in message.lower():
            raise RuntimeError(
                f"Groq model '{GROQ_MODEL_NAME}' was not found (likely deprecated/decommissioned "
                "by Groq -- their model catalog changes over time). Check the current list at "
                "https://console.groq.com/docs/models, then set the GROQ_MODEL_NAME environment "
                f"variable to override it without editing code. Run `python main.py --test-groq` "
                f"to verify. Original error: {message}"
            ) from e
        raise
    return response.choices[0].message.content


def _build_prompt(facets_batch: list[dict], conversation: str) -> str:
    """
    Build a focused prompt for ONE batch (<=10 facets). Includes each
    facet's scoring anchor so the model has a concrete rubric instead of
    guessing what "high Risktaking" means. Explicitly instructs the model to
    abstain rather than guess, and to output JSON only (no prose, no markdown
    fences) so parsing downstream is as robust as possible.
    """
    facet_lines = []
    for f in facets_batch:
        anchor = f.get("scoring_anchors", "").strip()
        facet_lines.append(f'- "{f["facet_normalized"]}" -- scale: {anchor}')
    facet_block = "\n".join(facet_lines)

    return f"""You are a careful, conservative personality-assessment scorer. You will read a short conversation and score ONLY the facets listed below, each on a 1-5 integer scale.

CONVERSATION:
\"\"\"
{conversation}
\"\"\"

FACETS TO SCORE (score ONLY these {len(facets_batch)} facets, nothing else):
{facet_block}

RULES:
1. If the conversation gives clear, direct evidence for a facet, set "status": "scored" and give an integer "score" from 1 to 5 using the scale provided.
2. If the conversation is too short, vague, ambiguous, or simply does not touch on a facet, set "status": "insufficient_evidence", "score": null. Do NOT guess or default to a middle score -- abstaining is the correct, expected answer for most facets in most short conversations.
3. Never invent facts not present in the conversation. Base every score strictly on what was said.
4. Respond with VALID JSON ONLY -- a single JSON array, no markdown code fences, no explanation text before or after.
5. Each array element must have exactly this shape:
{{"facet": "<facet name exactly as given>", "score": <1-5 or null>, "status": "scored" | "insufficient_evidence" | "not_observable", "confidence": "high" | "medium" | "low", "evidence": "<one sentence citing what in the conversation justifies this, or why there is insufficient evidence>"}}

Return the JSON array now, for all {len(facets_batch)} facets, in the same order given above."""


def _extract_json_array(raw_text: str) -> Any:
    """
    Robustly pull a JSON array out of an LLM response. Handles the common
    failure modes: markdown code fences, leading/trailing prose, or the
    model wrapping the array in an object like {"results": [...]}.
    Raises json.JSONDecodeError / ValueError if nothing usable is found --
    callers must catch this and mark the batch as parse_error rather than
    crash the whole pipeline.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to grabbing the first [...] block in the text.
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not bracket_match:
            raise ValueError("No JSON array found in model output.")
        parsed = json.loads(bracket_match.group(0))

    # Some models wrap the array: {"results": [...]} or {"facets": [...]}
    if isinstance(parsed, dict):
        for key in ("results", "facets", "data", "scores"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        raise ValueError(f"Expected a JSON array, got an object with keys: {list(parsed.keys())}")

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got {type(parsed)}")

    return parsed


def _sanitize_result(item: dict, expected_facet_names: set[str]) -> dict:
    """
    Validate/coerce one parsed result item into our canonical schema.
    Anything malformed (missing fields, out-of-range score, unknown status)
    is corrected conservatively -- we never let a bad field silently produce
    a false-confidence scored result.
    """
    facet_name = str(item.get("facet", "")).strip()

    status = item.get("status", "insufficient_evidence")
    if status not in VALID_STATUSES:
        status = "insufficient_evidence"

    score = item.get("score", None)
    if status != "scored":
        score = None
    else:
        try:
            score = int(score)
            if score < 1 or score > 5:
                score = None
                status = "insufficient_evidence"
        except (TypeError, ValueError):
            score = None
            status = "insufficient_evidence"

    confidence = item.get("confidence", "low")
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"

    evidence = str(item.get("evidence", "")).strip() or "No evidence provided by model."

    return {
        "facet": facet_name,
        "score": score,
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
    }


def score_facet_batch(facets_batch: list[dict], conversation: str) -> list[dict]:
    """
    Score ONE batch of up to BATCH_SIZE facets against a conversation.

    facets_batch: list of facet metadata dicts (must include
                  'facet_normalized', 'scoring_anchors', 'category').
    conversation: the raw conversation text.

    Returns a list of result dicts, one per input facet, ALWAYS the same
    length as facets_batch (missing/unparseable facets are filled in as
    parse_error placeholders so callers can rely on 1:1 correspondence).
    """
    if not facets_batch:
        return []
    if len(facets_batch) > BATCH_SIZE:
        raise ValueError(f"Batch of {len(facets_batch)} exceeds max BATCH_SIZE={BATCH_SIZE}.")

    # --- Safety gate: never let medical/biological facets reach the LLM for
    # scoring, even if they slipped through retrieval somehow. Force abstain.
    safe_batch = []
    forced_results = {}
    for f in facets_batch:
        if f.get("category") == "medical_biological":
            forced_results[f["facet_normalized"]] = {
                "facet": f["facet_normalized"],
                "score": None,
                "status": "not_observable",
                "confidence": "high",
                "evidence": "Medical/biological facets are never scored from conversation text (safety rule).",
            }
        else:
            safe_batch.append(f)

    results_by_name: dict[str, dict] = dict(forced_results)

    if safe_batch:
        prompt = _build_prompt(safe_batch, conversation)
        expected_names = {f["facet_normalized"] for f in safe_batch}

        try:
            raw_text = _call_llm(prompt)
        except Exception as e:
            # Neither backend available, Ollama connection refused, Groq API
            # error, etc. -- see _call_llm()/detect_backend(). Mark every
            # facet in this sub-batch as a parse_error rather than crash.
            for f in safe_batch:
                results_by_name[f["facet_normalized"]] = {
                    "facet": f["facet_normalized"],
                    "score": None,
                    "status": "parse_error",
                    "confidence": "low",
                    "evidence": f"LLM call failed: {e}",
                }
            raw_text = None

        if raw_text is not None:
            try:
                parsed_items = _extract_json_array(raw_text)
                for item in parsed_items:
                    if not isinstance(item, dict):
                        continue
                    sanitized = _sanitize_result(item, expected_names)
                    if sanitized["facet"] in expected_names:
                        # Extra safety net: if the model somehow scored a medical
                        # facet, force abstention rather than trust it.
                        results_by_name[sanitized["facet"]] = sanitized
            except (json.JSONDecodeError, ValueError) as e:
                for f in safe_batch:
                    if f["facet_normalized"] not in results_by_name:
                        results_by_name[f["facet_normalized"]] = {
                            "facet": f["facet_normalized"],
                            "score": None,
                            "status": "parse_error",
                            "confidence": "low",
                            "evidence": f"Could not parse model output as JSON: {e}",
                        }

    # Ensure every requested facet has a result, in the original order.
    final_results = []
    for f in facets_batch:
        name = f["facet_normalized"]
        if name in results_by_name:
            final_results.append(results_by_name[name])
        else:
            final_results.append({
                "facet": name,
                "score": None,
                "status": "parse_error",
                "confidence": "low",
                "evidence": "Model did not return a result for this facet.",
            })
    return final_results


def score_facets(facets: list[dict], conversation: str, batch_size: int = BATCH_SIZE) -> list[dict]:
    """Convenience wrapper: splits an arbitrary-length facet list into
    batches of `batch_size` and scores each, concatenating results."""
    all_results = []
    for i in range(0, len(facets), batch_size):
        batch = facets[i:i + batch_size]
        all_results.extend(score_facet_batch(batch, conversation))
    return all_results
