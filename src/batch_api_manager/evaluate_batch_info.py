import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
from openai import OpenAI

DEFAULT_PRICING: Dict[str, Tuple[float, float, float]] = {
    "gpt-5-2025-08-07": (1.25, 0.125, 10.0),
    "gpt-4o": (2.5, 1.25, 10.0),
}

def parse_model_pricing(overrides: list) -> Dict[str, Tuple[float, float, float]]:
    """
    Parse custom model pricing overrides of the form: model=IN/CACHED/OUT (per 1M)
    Example: gpt-5=1.25/0.125/10.0
    """
    pricing = dict(DEFAULT_PRICING)
    for item in overrides or []:
        if "=" not in item or item.count("/") != 2:
            raise ValueError(f"Bad --model-pricing entry: {item}. Expected model=IN/CACHED/OUT")
        model, rest = item.split("=", 1)
        inp, cached, outp = rest.split("/")
        pricing[model.strip()] = (float(inp), float(cached), float(outp))
    return pricing

def cents_per_token(per_million: float) -> float:
    # Convert price per 1M tokens in USD to price per token in USD
    return per_million / 1_000_000.0

def safe_get(d, *keys, default=None):
    """
    Safely retrieve nested values from dicts, Pydantic models, or objects via
    a sequence of keys/attributes. Falls back to `default` if any step is missing.

    This supports:
    - dict-like access: cur.get(k)
    - Pydantic v2 models: cur.model_dump().get(k)
    - Pydantic v1 models: cur.dict().get(k)
    - attribute access: getattr(cur, k)
    """
    cur = d
    for k in keys:
        if cur is None:
            return default
        # Direct dict access
        if isinstance(cur, dict):
            cur = cur.get(k)
            continue

        # Pydantic v2 model_dump or v1 dict
        mapping = None
        try:
            if hasattr(cur, "model_dump") and callable(getattr(cur, "model_dump")):
                mapping = cur.model_dump()
            elif hasattr(cur, "dict") and callable(getattr(cur, "dict")):
                mapping = cur.dict()
        except Exception:
            mapping = None

        if isinstance(mapping, dict) and k in mapping:
            cur = mapping.get(k)
            continue

        # Attribute access as a last resort
        try:
            cur = getattr(cur, k)
        except Exception:
            return default

    return default if cur is None else cur

def iso_utc(ts: Optional[int]) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return ""

def compute_cost_usd(model: str, usage: dict, pricing: Dict[str, Tuple[float, float, float]]) -> Tuple[float, int, int, int]:
    """
    Compute total cost in USD given usage and pricing table.
    Returns (total_cost_usd, input_tokens, cached_input_tokens, output_tokens).
    """
    input_tokens = safe_get(usage, "input_tokens", default=0) or 0
    cached_tokens = safe_get(usage, "input_tokens_details", "cached_tokens", default=0) or 0
    output_tokens = safe_get(usage, "output_tokens", default=0) or 0

    # Resolve model key in pricing (exact match preferred; else try prefix match like "gpt-5-2025-08-07" -> "gpt-5")
    key = model
    if key not in pricing:
        # naive fallback: try primary family key up to first space or first @date suffix
        for cand in pricing.keys():
            if key.startswith(cand):
                key = cand
                break

    in_per_m, cached_per_m, out_per_m = pricing.get(key, pricing.get("gpt-5", (1.25, 0.125, 10.0)))

    # Non-cached input tokens are billed at input rate
    non_cached_input = max(0, int(input_tokens) - int(cached_tokens))

    cost_input = non_cached_input * cents_per_token(in_per_m)
    cost_cached = cached_tokens * cents_per_token(cached_per_m)
    cost_output = output_tokens * cents_per_token(out_per_m)

    total_cost = cost_input + cost_cached + cost_output
    return total_cost, int(input_tokens), int(cached_tokens), int(output_tokens)


def compute_time_ms(created_at: Optional[int], in_progress_at: Optional[int], completed_at: Optional[int]) -> Tuple[int, int]:
    """Compute overall time costs (in ms) for a batch.

    Returns (time_total_ms, time_run_ms):
      - time_total_ms: from created_at -> completed_at
      - time_run_ms:   from in_progress_at -> completed_at (approximate compute window)
    Any missing timestamps result in 0 for that metric.
    """
    def ms(a: Optional[int], b: Optional[int]) -> int:
        try:
            if a is None or b is None:
                return 0
            return int(max(0, (b - a) * 1000))
        except Exception:
            return 0

    total_ms = ms(created_at, completed_at)
    run_ms = ms(in_progress_at, completed_at)
    return total_ms, run_ms

def extract_experiment_from_input_path(input_path: Optional[str]) -> Tuple[str, str, str]:
    """
    Derive (dataset, shots, experiment) from the log line's input_path.
    Examples:
      /.../logs/pubmed/gpt5/generation_4bit.jsonl -> ("pubmed", "none", "pubmed_none")
      /.../logs/few_shot/pubmed/gpt5/generation_4bit_fewshot_5.jsonl -> ("pubmed", "5", "pubmed_5")

    Fallbacks to ("", "none", "") if not determinable.
    """
    if not input_path or not isinstance(input_path, str):
        return "", "none", ""

    p = input_path
    parts = p.split(os.sep)

    dataset = ""
    try:
        if "logs" in parts:
            i = parts.index("logs")
            # logs/<dataset>/... or logs/few_shot/<dataset>/...
            if i + 1 < len(parts) and parts[i + 1] == "few_shot":
                if i + 2 < len(parts):
                    dataset = parts[i + 2]
            elif i + 1 < len(parts):
                dataset = parts[i + 1]
    except Exception:
        dataset = ""

    # shots: default none, detect from filename tokens like fewshot_5
    shots = "none"
    try:
        filename = parts[-1].lower()
        m = re.search(r"few\s*shot[_-]?(\d+)", filename.replace(" ", ""))
        if not m:
            # also scan full path for safety
            m = re.search(r"few\s*shot[_-]?(\d+)", p.replace(" ", "").lower())
        if m:
            shots = m.group(1)
        elif "few_shot" in parts:
            # few_shot folder but no explicit number; keep default unless we can infer later
            shots = shots  # no-op keeps "none"
    except Exception:
        shots = "none"

    experiment = f"{dataset}_{shots}" if dataset else ""
    return dataset, shots, experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to log file with lines containing batch_id")
    ap.add_argument("--out", default="batch_costs.csv", help="Output CSV path")
    ap.add_argument("--model-pricing", nargs="*", help="Override pricing entries: model=IN/CACHED/OUT (per 1M)")
    ap.add_argument("--include-failed", action="store_true", help="Use total requests (completed+failed) for $/100 denominator (default uses completed only)")
    args = ap.parse_args()

    pricing = parse_model_pricing(args.model_pricing)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(2)

    client = OpenAI()

    # Collect unique batch_ids from the log file and map to experiment derived from input_path
    batch_ids = []
    batch_to_experiment: Dict[str, str] = {}
    with open(args.logfile, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    start = line.index("{")
                    end = line.rindex("}") + 1
                    obj = json.loads(line[start:end])
                except Exception:
                    print(f"WARNING: Skipping non-JSON line {line_num}", file=sys.stderr)
                    continue
            batch_id = obj.get("batch_id") or obj.get("id") or obj.get("batchId")
            if batch_id and isinstance(batch_id, str) and batch_id.startswith("batch_"):
                batch_ids.append(batch_id)
                # Derive experiment once per batch_id (first occurrence wins)
                if batch_id not in batch_to_experiment:
                    input_path = obj.get("input_path")
                    _, _, exp = extract_experiment_from_input_path(input_path)
                    if exp:
                        batch_to_experiment[batch_id] = exp

    if not batch_ids:
        print("No batch_ids found in the logfile.", file=sys.stderr)
        sys.exit(1)

    # De-duplicate while preserving order
    seen = set()
    uniq_batch_ids = []
    for b in batch_ids:
        if b not in seen:
            uniq_batch_ids.append(b)
            seen.add(b)

    rows = []
    for b in uniq_batch_ids:
        try:
            batch = client.batches.retrieve(b)
        except Exception as e:
            print(f"ERROR: Failed to retrieve {b}: {e}", file=sys.stderr)
            continue

        model = getattr(batch, "model", None) or safe_get(batch, "model", default="")
        usage = getattr(batch, "usage", None) or safe_get(batch, "usage", default={})
        req_counts = getattr(batch, "request_counts", None) or safe_get(batch, "request_counts", default={})

        completed = int(safe_get(req_counts, "completed", default=0) or 0)
        failed = int(safe_get(req_counts, "failed", default=0) or 0)
        total = int(safe_get(req_counts, "total", default=completed + failed) or (completed + failed))

        total_cost, input_tokens, cached_tokens, output_tokens = compute_cost_usd(model, usage, pricing)

        denom = total if args.include_failed else (completed if completed > 0 else total)
        cost_per_100 = (total_cost / denom * 100.0) if denom > 0 else 0.0

        created_at = getattr(batch, "created_at", None) or safe_get(batch, "created_at", default=None)
        in_progress_at = getattr(batch, "in_progress_at", None) or safe_get(batch, "in_progress_at", default=None)
        completed_at = getattr(batch, "completed_at", None) or safe_get(batch, "completed_at", default=None)
        status = getattr(batch, "status", None) or safe_get(batch, "status", default="")

        time_total_ms, time_run_ms = compute_time_ms(created_at, in_progress_at, completed_at)

        rows.append({
            "experiment": batch_to_experiment.get(b, ""),
            "batch_id": b,
            "model": model,
            # "status": status,
            # "created_at": iso_utc(created_at),
            # "completed_at": iso_utc(completed_at),
            "requests_completed": completed,
            "requests_failed": failed,
            "requests_total": total,
            "tokens_in": input_tokens,
            # "tokens_in_cached": cached_tokens,
            "tokens_out": output_tokens,
            "cost_total_usd": round(total_cost, 6),
            "cost_per_100_completed_usd": round(cost_per_100, 6),
            "time_total_ms": time_total_ms,
            "time_run_ms": time_run_ms,
        })

    # Write CSV
    out_path = args.out
    fieldnames = [
        "experiment",  # first column acts as index-like identifier
        "batch_id", 
        "model",
        # "status",
        # "created_at",
        # "completed_at",
        "requests_completed",
        "requests_failed",
        "requests_total",
        "tokens_in",
        # "tokens_in_cached",
        "tokens_out",
        "cost_total_usd", "cost_per_100_completed_usd",
        "time_total_ms", "time_run_ms",
        ]

    with open(out_path, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if rows:
        # Show a quick summary
        total_cost = sum(r["cost_total_usd"] for r in rows)
        total_completed = sum(r["requests_completed"] for r in rows)
        total_in = sum(r["tokens_in"] for r in rows)
        total_out = sum(r["tokens_out"] for r in rows)
        total_time_ms = sum(r.get("time_total_ms", 0) for r in rows)
        avg_run_ms = (sum(r.get("time_run_ms", 0) for r in rows) / len(rows)) if rows else 0
        print(
            f"Total cost: ${total_cost:.4f} | Tokens In: {total_in:,} | Tokens Out: {total_out:,} | "
            f"Avg $/100 completed: {(total_cost / max(1,total_completed) * 100):.6f} | "
            f"Total time: {total_time_ms:,} ms | Avg run time: {int(avg_run_ms):,} ms"
        )

if __name__ == "__main__":
    main()
