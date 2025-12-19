import os, time, re, json, math, argparse, random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import difflib
try:
    from openai import OpenAI
    _NEW_OPENAI_SDK = True
except Exception:
    import openai as OpenAI  
    _NEW_OPENAI_SDK = False

OPENAI_API_KEY = "xx" 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



client = None
if _NEW_OPENAI_SDK:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    OpenAI.api_key = OPENAI_API_KEY
    client = OpenAI  

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SPLIT_DEFAULT = "test"
TIMEOUT_S = 120

CASE_TRUNC_1 = 900   
CASE_TRUNC_2 = 500   
SHOT_TRUNC   = 400   

G5_OUT_1 = 2000
G5_OUT_2 = 5000

CHAT_OUT  = 320

PRICING_PER_1K = {  
    "gpt-5":  {"input": 0.005,  "output": 0.015},
    "gpt-5-chat-latest": {"input": 0.005, "output": 0.015},  
    "gpt-4o": {"input": 0.0025, "output": 0.010},
}

REFUSAL_PATTERNS = [
    r"\bi\s+(?:cannot|can't)\s+answer\b",
    r"\bi\s+will\s+not\s+provide\b",
    r"\bi\s+refuse\b",
    r"\bunable\s+to\s+answer\b",
]
REFUSAL_WHITELIST = [
    r"not (?:intended|a substitute) for medical advice",
    r"informational purposes only",
    r"consult a (?:doctor|physician|professional)",
]

def is_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(re.search(p, t) for p in REFUSAL_WHITELIST):
        return False
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)

@dataclass
class ItemLog:
    qid: str
    model: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    gold: str
    pred: str
    is_correct_ce: int
    hit_tau: int
    hit_alias: int
    refusal: int
    err_missing: int
    raw_output: str

def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens / 1000.0) * p["input"] + (out_tokens / 1000.0) * p["output"]

def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\s\-_\/]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = s.replace("ami", "acute myocardial infarction")
    s = s.replace("mi", "myocardial infarction")
    s = re.sub(r"\b(type|acute|chronic|unspecified)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def contains_alias_like(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    A, B = set(a.split()), set(b.split())
    short, long = (A, B) if len(A) <= len(B) else (B, A)
    return len(short & long) >= max(1, int(0.8 * len(short)))

def _extract_text_from_responses(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out_list = getattr(resp, "output", None) or getattr(resp, "outputs", None) or []
    def _get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    buf = []
    for item in out_list:
        content_list = _get(item, "content", []) or []
        for c in content_list:
            ctype = _get(c, "type", None)
            if ctype in (None, "output_text", "input_text", "text"):
                t = _get(c, "text", None)
                if isinstance(t, str) and t.strip():
                    buf.append(t.strip())
                elif t is not None:
                    val = _get(t, "value", None)
                    if isinstance(val, str) and val.strip():
                        buf.append(val.strip())
                else:
                    s = _get(c, "string", None) or _get(c, "content", None)
                    if isinstance(s, str) and s.strip():
                        buf.append(s.strip())
    return "\n".join(buf).strip()

def parse_json_line(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return {}
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        frag2 = re.sub(r"(\w+)\s*:", r'"\1":', frag)
        try:
            return json.loads(frag2)
        except Exception:
            return {}

def build_kshot_block(df: pd.DataFrame, k: int, exclude_row_idx: int) -> str:
    if k <= 0 or len(df) <= 1:
        return ""
    indices = list(range(len(df)))
    if exclude_row_idx in indices:
        indices.remove(exclude_row_idx)
    if not indices:
        return ""

    random.shuffle(indices)
    picks = indices[:k]
    lines: List[str] = ["Examples (format only, not the target):"]
    for rank, j in enumerate(picks, start=1):
        ex_case = str(df.iloc[j]["case_text"])
        ex_gold = str(df.iloc[j]["gold_final"])
        if len(ex_case) > SHOT_TRUNC:
            ex_case = ex_case[:SHOT_TRUNC]
        lines.append(
            f"Example #{rank} — Patient case:\n{ex_case}\n"
            "One-line JSON:\n"
            '{"diagnostic_reasoning":"brief clinical rationale",'
            f'"final_diagnosis":"{ex_gold}"\n'
        )
    return "\n".join(lines).strip()

def build_prompt_mcr(case_text: str, kshot_block: str) -> Tuple[str, str]:
    system = (
        "You are participating in an academic benchmark evaluation for clinical reasoning. "
        "This is NOT medical advice. Return ONLY a single-line JSON."
    )
    case = case_text[:CASE_TRUNC_1] if len(case_text) > CASE_TRUNC_1 else case_text
    few = (kshot_block + "\n\n") if kshot_block else ""
    user = f"""\
{few}Patient case (abbreviated for evaluation):
{case}

Immediately output EXACTLY one line JSON (do this first):
{{"diagnostic_reasoning": "<<=80 words>>", "final_diagnosis": "<concise primary diagnosis>"}}

Rules:
- No extra text before/after the JSON.
- Do NOT deliberate step-by-step in visible text.
- Focus on the most likely single primary diagnosis.
"""
    return system, user

def _concat_case_fields(ex: Dict[str, Any]) -> str:
    skip_keys = {"id", "label", "labels", "gold", "final_diagnosis", "diagnosis", "gt", "answer",
                 "Right Option", "Options"}
    parts = []
    for k, v in ex.items():
        if k in skip_keys:
            continue
        if isinstance(v, str):
            vv = v.strip()
            if vv:
                parts.append(f"{k.replace('_', ' ').title()}: {vv}")
    case_text = "\n".join(parts).strip()
    if len(case_text) > CASE_TRUNC_1:
        case_text = case_text[:CASE_TRUNC_1]
    return case_text

def _pick_gold_field(ex: Dict[str, Any]) -> str:
    candidates = ["final_diagnosis", "diagnosis", "gold", "label", "gt_diagnosis", "answer"]
    for c in candidates:
        v = ex.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    diag = ex.get("labels") or ex.get("meta")
    if isinstance(diag, dict):
        for c in ["final_diagnosis", "diagnosis", "gold", "label"]:
            v = diag.get(c)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def load_medcasereasoning(split: str = "test", max_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    ds = load_dataset("zou-lab/MedCaseReasoning", split=split)
    print(f"[INFO] Loaded split='{split}' with {len(ds)} rows.")
    records = []
    for i, ex in enumerate(ds):
        qid = str(ex.get("id", i))
        gold = _pick_gold_field(ex)
        case_text = _concat_case_fields(ex)
        if not case_text or not gold:
            continue
        records.append({"id": qid, "case_text": case_text, "gold_final": gold})
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Parsed DataFrame is empty. Schema may have changed; update the loader rules.")
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df

def call_gpt(system_prompt: str, user_prompt: str, model: str) -> Dict[str, Any]:
    """
    Router:
      - gpt-5: Responses API
      - gpt-5-chat* / gpt-4o: Chat Completions
    """
    if model == "gpt-5":
        return _call_gpt5(system_prompt, user_prompt)
    else:
        return _call_chat_models(system_prompt, user_prompt, model)

def _call_gpt5(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    start = time.time()

    def _once(sp, up, max_out, note):
        try:
            resp = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": sp},
                    {"role": "user",   "content": up},
                ],
                max_output_tokens=max_out,
                reasoning={"effort": "low"},
            )
            text = _extract_text_from_responses(resp)
            usage = getattr(resp, "usage", None) or {}
            in_tok = getattr(usage, "input_tokens", 0)
            out_tok = getattr(usage, "output_tokens", 0)
            status = getattr(resp, "status", None)
            if not text:
                print(f"[WARN] gpt-5 empty text ({note}). status={status} in={in_tok} out={out_tok}")
            return text, in_tok, out_tok, status, None
        except Exception as e:
            return "", 0, 0, None, str(e)

    text, in_tok, out_tok, status, err = _once(system_prompt, user_prompt, max_out=G5_OUT_1, note="try1")
    if text:
        latency_ms = (time.time() - start) * 1000.0
        return {"ok": True, "text": text, "latency_ms": latency_ms,
                "input_tokens": in_tok, "output_tokens": out_tok, "status": status, "error": None}

    case_match = re.search(r"Patient case.*?:\s*(.+?)\n\nImmediately output EXACTLY one line JSON", user_prompt, flags=re.S)
    if case_match:
        case_body = case_match.group(1)
        if len(case_body) > CASE_TRUNC_2:
            case_body2 = case_body[:CASE_TRUNC_2]
            user_prompt2 = re.sub(re.escape(case_body), case_body2, user_prompt, count=1)
        else:
            user_prompt2 = user_prompt
    else:
        user_prompt2 = user_prompt[:1800]

    text, in_tok, out_tok, status, err = _once(system_prompt, user_prompt2, max_out=G5_OUT_2, note="try2-tight")
    latency_ms = (time.time() - start) * 1000.0
    return {"ok": bool(text), "text": text or "", "latency_ms": latency_ms,
            "input_tokens": in_tok, "output_tokens": out_tok, "status": status,
            "error": None if text else (err or "empty_text")}

def _call_chat_models(system_prompt: str, user_prompt: str, model: str) -> Dict[str, Any]:
    start = time.time()
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "timeout": TIMEOUT_S,
            "max_completion_tokens": CHAT_OUT,
        }
        if model.startswith("gpt-4o"):
            kwargs["temperature"] = 0

        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None) or {}
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)
        latency_ms = (time.time() - start) * 1000.0
        if not text:
            print(f"[WARN] {model} empty text. in={in_tok} out={out_tok}")
        return {"ok": bool(text), "text": text or "", "latency_ms": latency_ms,
                "input_tokens": in_tok, "output_tokens": out_tok, "status": None, "error": None}
    except Exception as e:
        latency_ms = (time.time() - start) * 1000.0
        print(f"[ERROR call_chat {model}] {e}")
        return {"ok": False, "text": "", "latency_ms": latency_ms,
                "input_tokens": 0, "output_tokens": 0, "status": None, "error": str(e)}

def eval_model_on_mcr(model_name: str, df: pd.DataFrame, kshot: int = 0, debug: bool = False) -> Tuple[pd.DataFrame, dict]:
    logs: List[ItemLog] = []
    indices = list(range(len(df)))
    if debug:
        n = min(args.n, len(df))
        indices = random.sample(indices, n)

    os.makedirs("results", exist_ok=True)
    if debug:
        open("results/_debug_raw.txt", "w", encoding="utf-8").close()

    for row_idx in tqdm(indices, total=len(indices), desc=f"Evaluating {model_name} (Acc@CE/τ/ALIAS)"):
        row = df.iloc[row_idx]
        case_text = row["case_text"]
        gold_final = row["gold_final"]

        kshot_block = build_kshot_block(df, kshot, exclude_row_idx=row_idx)

        sys_p, usr_p = build_prompt_mcr(case_text, kshot_block)

        resp = call_gpt(sys_p, usr_p, model=model_name)

        if not resp.get("ok", False):
            print(f"[API-ERR] model={model_name} qid={row['id']} err={resp.get('error')}", flush=True)
            text = ""
            pred_diag = ""
            pred_norm = ""
            gold_norm = norm(gold_final)
            hit_ce = 0
            hit_tau = 0
            hit_alias = 0
            refusal = 0
            err_missing = 1
        else:
            text = resp["text"]

            if debug:
                with open("results/_debug_raw.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n==== [{model_name}] QID {row['id']} ====\n{text}\n")

            pred_json = parse_json_line(text)
            pred_diag = pred_json.get("final_diagnosis", "") if isinstance(pred_json, dict) else ""
            pred_norm = norm(pred_diag)
            gold_norm = norm(gold_final)

            hit_ce = int(pred_norm == gold_norm and bool(pred_norm))
            hit_tau = int(jaccard(pred_norm, gold_norm) >= 0.8 and bool(pred_norm))
            hit_alias = int(contains_alias_like(pred_norm, gold_norm) and bool(pred_norm))
            refusal = 1 if is_refusal(text) else 0
            err_missing = 1 if not pred_norm else 0

        cost = estimate_cost_usd(model_name, resp.get("input_tokens", 0), resp.get("output_tokens", 0))
        logs.append(ItemLog(
            qid=row["id"], model=model_name, latency_ms=resp.get("latency_ms", 0.0),
            tokens_in=resp.get("input_tokens", 0), tokens_out=resp.get("output_tokens", 0), cost_usd=cost,
            gold=gold_final, pred=pred_diag, is_correct_ce=hit_ce, hit_tau=hit_tau, hit_alias=hit_alias,
            refusal=refusal, err_missing=err_missing, raw_output=text
        ))

    df_log = pd.DataFrame([asdict(x) for x in logs])
    summary = {
        "model": model_name,
        "n": len(df_log),
        "acc_ce": round(df_log["is_correct_ce"].mean(), 4) if len(df_log) else 0.0,
        "acc_tau": round(df_log["hit_tau"].mean(), 4) if len(df_log) else 0.0,
        "acc_alias": round(df_log["hit_alias"].mean(), 4) if len(df_log) else 0.0,
        "avg_tokens_in": round(df_log["tokens_in"].mean(), 2) if len(df_log) else 0.0,
        "avg_tokens_out": round(df_log["tokens_out"].mean(), 2) if len(df_log) else 0.0,
        "total_cost_usd": round(df_log["cost_usd"].sum(), 4) if len(df_log) else 0.0,
        "avg_latency_ms": round(df_log["latency_ms"].mean(), 2) if len(df_log) else 0.0,
        "refusal_rate": round(df_log["refusal"].mean(), 4) if len(df_log) else 0.0,
    }
    return df_log, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=SPLIT_DEFAULT, help="HF split: test/dev/train")
    parser.add_argument("--model", default="both", choices=["gpt-5", "gpt-5-chat-latest", "gpt-4o", "both"])
    parser.add_argument("--kshot", type=int, default=0, help="few-shot number (0=zero-shot)")
    parser.add_argument("--debug", action="store_true", help="Sample mode with small N and raw dump.")
    parser.add_argument("--n", type=int, default=10, help="Sample size when --debug is on.")
    args = parser.parse_args()

    df = load_medcasereasoning(split=args.split)
    print(f"Loaded MedCaseReasoning {args.split} split with {len(df)} samples.")

    if args.model == "both":
        models = ["gpt-5", "gpt-4o"]  
    else:
        models = [args.model]

    summaries = []
    for m in models:
        df_log, summary = eval_model_on_mcr(m, df, kshot=args.kshot, debug=args.debug)
        suffix = "DEBUG" if args.debug else args.split
        out_csv = f"results/results_medcasereasoning_{suffix}_{m}_k{args.kshot}.csv"
        df_log.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote per-item logs to: {out_csv}")
        print("[RESULT]", summary)
        summaries.append(summary)

    suffix = "DEBUG" if args.debug else args.split
    out_json = f"results/summary_medcasereasoning_{m.lower()}_{args.split}_k{args.kshot}.json"
    with open(out_json, "w") as f:
        json.dump(summaries, f, indent=2)