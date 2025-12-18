import json
import os
import re
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


def _iter_jsonl(path: str) -> Iterable[dict]:
    """Yield JSON objects from a file where each line is a JSON object.

    Skips empty/invalid lines gracefully.
    """
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def _extract_prompt_from_request(obj: dict) -> Optional[str]:
    """Extract the user prompt/question string from a Batch request line.

    Tries common OpenAI request shapes used in this repo.
    """
    body = obj.get("body") if isinstance(obj, dict) else None
    if not isinstance(body, dict):
        return None

    # Prefer chat-style messages if present
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        # Join all message contents for robustness
        parts = []
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                # content may be a list of blocks
                for blk in c:
                    if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                        parts.append(blk["text"])
        if parts:
            return "\n".join(parts)

    # Fallbacks: some endpoints might use `input` or `prompt`
    for key in ("input", "prompt"):
        val = body.get(key)
        if isinstance(val, str):
            return val

    return None


def _extract_raw_output_from_response(obj: dict) -> Optional[str]:
    """Extract the model's text output from a Batch response line.

    Tries several shapes (/v1/chat/completions and /v1/responses outputs).
    """
    resp = obj.get("response") if isinstance(obj, dict) else None
    if not isinstance(resp, dict):
        return None
    body = resp.get("body")
    if not isinstance(body, dict):
        return None

    # Common chat completions shape
    try:
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    except Exception:
        pass

    # Newer /v1/responses outputs
    try:
        output = body.get("output")
        if isinstance(output, list) and output:
            # Typical: output[0].content[0].text
            first = output[0]
            if isinstance(first, dict):
                content_list = first.get("content")
                if isinstance(content_list, list) and content_list:
                    blk = content_list[0]
                    if isinstance(blk, dict):
                        if isinstance(blk.get("text"), str):
                            return blk["text"]
                        # Some SDKs nest another level
                        if isinstance(blk.get("content"), list) and blk["content"]:
                            inner = blk["content"][0]
                            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                                return inner["text"]
    except Exception:
        pass

    # Fallback: try top-level text
    if isinstance(body.get("text"), str):
        return body["text"]

    return None


def _extract_error_message(obj: dict) -> Optional[str]:
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str):
            return msg
        # Some error bodies put message directly in response.body.error.message
        body = err.get("body") if isinstance(err.get("body"), dict) else None
        if isinstance(body, dict):
            if isinstance(body.get("message"), str):
                return body["message"]
    # Try response.body.error.message pattern
    resp = obj.get("response") if isinstance(obj, dict) else None
    if isinstance(resp, dict):
        body = resp.get("body")
        if isinstance(body, dict):
            e2 = body.get("error")
            if isinstance(e2, dict) and isinstance(e2.get("message"), str):
                return e2["message"]
    return None


def _extract_usage_from_response(obj: dict) -> Dict[str, int]:
    """Extract token usage fields from a Batch response line.

    Supports both legacy /v1/chat/completions usage schema and newer /v1/responses
    schema. Returns a dict with the following normalized integer fields when present:
      - tokens_in: prompt/input tokens
      - tokens_out: completion/output tokens
      - total_tokens
      - cached_tokens: prompt/input cached tokens (if any)
      - reasoning_tokens: output reasoning tokens (if any)
      - prompt_tokens, completion_tokens (legacy names for visibility)

    Missing values are omitted from the dict; callers should default to 0.
    """
    out: Dict[str, int] = {}
    resp = obj.get("response") if isinstance(obj, dict) else None
    if not isinstance(resp, dict):
        return out
    body = resp.get("body")
    if not isinstance(body, dict):
        return out

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return out

    # Newer schema (responses): input_tokens/output_tokens
    if any(k in usage for k in ("input_tokens", "output_tokens")):
        try:
            if isinstance(usage.get("input_tokens"), int):
                out["tokens_in"] = int(usage["input_tokens"])
            if isinstance(usage.get("output_tokens"), int):
                out["tokens_out"] = int(usage["output_tokens"])
            if isinstance(usage.get("total_tokens"), int):
                out["total_tokens"] = int(usage["total_tokens"])
            # details
            det_in = usage.get("input_tokens_details")
            if isinstance(det_in, dict) and isinstance(det_in.get("cached_tokens"), int):
                out["cached_tokens"] = int(det_in["cached_tokens"])
            det_out = usage.get("output_tokens_details")
            if isinstance(det_out, dict) and isinstance(det_out.get("reasoning_tokens"), int):
                out["reasoning_tokens"] = int(det_out["reasoning_tokens"])
        except Exception:
            pass

    # Legacy schema (chat.completions): prompt_tokens/completion_tokens
    if any(k in usage for k in ("prompt_tokens", "completion_tokens")):
        try:
            if isinstance(usage.get("prompt_tokens"), int):
                out["prompt_tokens"] = int(usage["prompt_tokens"])
                # Map to normalized tokens_in if not already set
                out.setdefault("tokens_in", out["prompt_tokens"])
            if isinstance(usage.get("completion_tokens"), int):
                out["completion_tokens"] = int(usage["completion_tokens"])
                # Map to normalized tokens_out if not already set
                out.setdefault("tokens_out", out["completion_tokens"])
            if isinstance(usage.get("total_tokens"), int):
                out["total_tokens"] = int(usage["total_tokens"])
            det_prompt = usage.get("prompt_tokens_details")
            if isinstance(det_prompt, dict) and isinstance(det_prompt.get("cached_tokens"), int):
                out["cached_tokens"] = int(det_prompt["cached_tokens"])
            det_comp = usage.get("completion_tokens_details")
            if isinstance(det_comp, dict) and isinstance(det_comp.get("reasoning_tokens"), int):
                out["reasoning_tokens"] = int(det_comp["reasoning_tokens"])
        except Exception:
            pass

    return out


def build_unified_dataframe(result_json_path: str) -> pd.DataFrame:
    """Build a unified DataFrame keyed by custom_id from three files that share a basename:

    - requests (.jsonl): original requests with prompts
    - results  (.json): successful responses
    - errors   (.errors.json): error lines

    Columns: [custom_id, question, raw_output, error_message, tokens_in, tokens_out, total_tokens,
              cached_tokens, reasoning_tokens, prompt_tokens, completion_tokens]
    Index: custom_id
    Missing fields are filled with empty strings.
    """
    if not result_json_path:
        raise ValueError("result_json_path is required")

    base_dir = os.path.dirname(result_json_path)
    base_name = os.path.basename(result_json_path)

    # Derive aligned paths
    outputs_path = result_json_path
    errors_path = result_json_path.replace(".json", ".errors.json")
    requests_path = result_json_path.replace(".json", ".jsonl")

    # Collect maps keyed by custom_id
    questions: Dict[str, str] = {}
    outputs: Dict[str, str] = {}
    usages: Dict[str, Dict[str, int]] = {}
    errors: Dict[str, str] = {}

    # Requests -> question/prompt
    for obj in _iter_jsonl(requests_path) or []:
        cid = obj.get("custom_id")
        if not isinstance(cid, str):
            continue
        q = _extract_prompt_from_request(obj)
        if isinstance(q, str):
            questions[cid] = q

    # Success outputs
    for obj in _iter_jsonl(outputs_path) or []:
        cid = obj.get("custom_id")
        if not isinstance(cid, str):
            continue
        text = _extract_raw_output_from_response(obj)
        if isinstance(text, str):
            outputs[cid] = text
        # Token usage (if any)
        u = _extract_usage_from_response(obj)
        if u:
            usages[cid] = u

    # Error lines
    for obj in _iter_jsonl(errors_path) or []:
        cid = obj.get("custom_id")
        if not isinstance(cid, str):
            continue
        emsg = _extract_error_message(obj)
        if isinstance(emsg, str):
            errors[cid] = emsg

    # Union of all ids
    all_ids = sorted(set(list(questions.keys()) + list(outputs.keys()) + list(errors.keys())))

    rows = []
    for cid in all_ids:
        q = questions.get(cid, "")
        out = outputs.get(cid)
        err = errors.get(cid)
        # If an error exists, clear raw_output per requirement
        if err and out:
            out = ""
        # Defaults for usage fields
        u = usages.get(cid, {})
        row = {
            "custom_id": cid,
            "question": q,
            "raw_output": out or "",
            "error message": err or "",
            # Token usage (normalized); default to 0 if missing
            "tokens_in": int(u.get("tokens_in", 0)),
            "tokens_out": int(u.get("tokens_out", 0)),
            "total_tokens": int(u.get("total_tokens", 0)),
            "cached_tokens": int(u.get("cached_tokens", 0)),
            "reasoning_tokens": int(u.get("reasoning_tokens", 0)),
            # Also expose legacy fields when available for convenience
            "prompt_tokens": int(u.get("prompt_tokens", u.get("tokens_in", 0))),
            "completion_tokens": int(u.get("completion_tokens", u.get("tokens_out", 0))),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Set custom_id as index only
        df.set_index("custom_id", inplace=True)
    else:
        # Ensure the expected columns exist even if empty
        df = pd.DataFrame(columns=[
            "question",
            "raw_output",
            "error message",
            "tokens_in",
            "tokens_out",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "prompt_tokens",
            "completion_tokens",
        ])
        df.index.name = "custom_id"

    return df


def extract_input_segment(full_prompt: str) -> str:
    """Extract the user question segment from a full prompt template.

    Logic:
    1. Find the first occurrence of a heading that starts with '### Input'. This may be
       '### Input:', '### Input Text:', etc. Capture everything after the first colon
       following that heading.
    2. Stop if a line starting with '### Output' (case-insensitive) appears.
    3. If no '### Input' marker is found, return the whole prompt (fallback).

    We keep this utility lightweight and forgiving so changes in prompt wording do not
    break extraction.
    """
    if not isinstance(full_prompt, str) or not full_prompt.strip():
        return ""

    # Normalize newlines
    text = full_prompt

    # Locate the Input marker
    m = re.search(r"###\s*Input[^:]*:(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        # Some prompts may use '### Input Text:' already covered; if nothing, fallback
        return full_prompt.strip()

    remainder = m.group(1)
    # Split on next ### Output (or ### Output Text) marker if present
    out_split = re.split(r"###\s*Output[^:]*:?", remainder, maxsplit=1, flags=re.IGNORECASE)
    segment = out_split[0]
    return segment.strip()
