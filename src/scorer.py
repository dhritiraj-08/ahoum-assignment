"""
scorer.py
---------
Uses a local Ollama model (llama3.1) to score a batch of retrieved facets
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
"""

import json
import re
from typing import Any

try:
    import ollama
except ImportError as e:
    raise ImportError("ollama package not installed. Run: pip install -r requirements.txt") from e

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1"
BATCH_SIZE = 10

VALID_STATUSES = {"scored", "insufficient_evidence", "not_observable"}
VALID_CONFIDENCE = {"high", "medium", "low"}

_client = ollama.Client(host=OLLAMA_HOST)


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
            response = _client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},  # low temperature: we want consistent, conservative scoring
            )
            raw_text = response["message"]["content"]
        except Exception as e:
            # Ollama not running, model not pulled, connection refused, etc.
            # Mark every facet in this sub-batch as a parse_error rather than crash.
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
