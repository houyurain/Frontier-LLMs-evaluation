import os, time, re, json, argparse, random 
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Tuple
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

DA_SPLIT = "test"
TIMEOUT_S = 60
MAX_COMPLETION_TOKENS = 256  

MODELS = ["gpt-5", "gpt-5-chat-latest", "gpt-4o"]

PRICING_PER_1K = {
    "gpt-5": {"input": 0.005, "output": 0.015},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-5-chat-latest": {"input": 0.005, "output": 0.015},  
}

REFUSAL_PATTERNS = [
    r"\bi\s+cannot\s+answer\b",
    r"\bi\s+can't\s+answer\b",
    r"\bunable\s+to\s+answer\b",
    r"\bi\s+will\s+not\s+provide\b",
    r"\bi\s+refuse\b",
]

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
    is_correct: int
    refusal: int
    err_missing: int
    err_inconsistent: int
    err_hallucinated: int
    raw_output: str

def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens / 1000.0) * p["input"] + (out_tokens / 1000.0) * p["output"]

def is_refusal(text: str) -> bool:
    if not text:
        return False  
    return any(re.search(pat, text.lower()) for pat in REFUSAL_PATTERNS)

def extract_choice_strict(text: str, allowed_letters: Tuple[str, ...]) -> str:
    """
    严格抽取最终选项字母。只认：
      - Final Answer: X
      - Answer: X / Choice: X
      - Correct answer is X
    若最后一行只有单个字母，也接受（在 allowed_letters 内）。
    """
    if not text:
        return ""
    patterns = [
        r"^\s*final\s*answer\s*[:：]?\s*\(?\s*([A-J])\s*\)?\s*$",
        r"^\s*(?:answer|choice)\s*[:：]?\s*\(?\s*([A-J])\s*\)?\s*$",
        r"^\s*correct\s*answer\s*(?:is|:)?\s*\(?\s*([A-J])\s*\)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.M)
        if m:
            L = m.group(1).upper()
            return L if L in allowed_letters else ""
    last_line = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.search(r"\b([A-J])\b\s*$", last_line)
    if m:
        L = m.group(1).upper()
        return L if L in allowed_letters else ""
    return ""

OPTION_LETTER_CTX = re.compile(
    r"(?ix)\b(?:final\s*answer|correct\s*answer|answer\s*(?:is|:))\b[^A-Z]*\(?\s*([A-J])\s*\)?"
)

def classify_errors(pred_choice: str, gold: str, valid_letters: Tuple[str, ...], model_text: str, is_correct: int):
    """
    返回 (missing, inconsistent, hallucinated)，互斥。
      - is_correct == 1 -> (0,0,0)
      - 优先级：Missing > Invalid label(算 inconsistent) > Hallucinated(文本出现矛盾多选) > Inconsistent
    """
    if is_correct == 1:
        return 0, 0, 0
    if not pred_choice:
        return 1, 0, 0
    if pred_choice not in valid_letters:
        return 0, 1, 0
    letters_ctx = {m.group(1).upper() for m in OPTION_LETTER_CTX.finditer(model_text or "")}
    contradictions = {l for l in letters_ctx if l in valid_letters and l != pred_choice}
    if len(contradictions) >= 1:
        return 0, 0, 1
    if pred_choice != gold:
        return 0, 1, 0
    return 0, 0, 0

def _format_one_example(question: str, options: List[str], letters: List[str], gold: str) -> str:
    labeled = [f"{letters[i]}. {options[i]}" for i in range(len(letters))]
    return (
        "Example:\n"
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{os.linesep.join(labeled)}\n"
        f"Final Answer: {gold}\n"
        "----\n"
    )

def build_fewshot_block(df: pd.DataFrame, kshot: int, exclude_qid: str) -> str:
    """从 df 中抽 k 条作为示例，避开当前题（exclude_qid）。如果不足 k，抽到有多少给多少。"""
    if kshot <= 0:
        return ""
    pool = df[df["id"] != exclude_qid]
    if pool.empty:
        return ""
    k = min(kshot, len(pool))
    shots = pool.sample(n=k, random_state=RANDOM_SEED).to_dict(orient="records")
    parts = []
    for ex in shots:
        question = ex["question"]
        options = ex["options"]
        letters = ex["letters"]
        gold = ex["gold"]
        parts.append(_format_one_example(question, options, letters, gold))
    return "".join(parts)

def build_prompt_da(question: str, options: List[str], letters: List[str],
                    model_name: str, fewshot_block: str = "") -> Tuple[str, str]:
    labeled = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    prefix = ""
    if fewshot_block:
        prefix = fewshot_block + "\n"

    if model_name.startswith("gpt-5-chat"):
        system = (
            "You are taking part in an academic benchmark evaluation. "
            "This is NOT medical advice. Output ONLY the final answer letter."
        )
        user = (
            f"{prefix}"
            f"Clinical Case (for research evaluation only):\n"
            f"{question.strip()}\n\n"
            f"Options:\n{os.linesep.join(labeled)}\n\n"
            "Rules (CRITICAL):\n"
            "- Output EXACTLY ONE line.\n"
            "- The line must be: Final Answer: <LETTER>\n"
            "- No explanation. No extra words.\n"
            f"- Allowed letters: {'/'.join(letters)}\n"
        )
    else:
        system = "You are a board-certified physician. Answer differential diagnosis MCQs correctly."
        user = (
            f"{prefix}"
            f"Clinical Case:\n{question.strip()}\n\n"
            f"Options:\n{os.linesep.join(labeled)}\n\n"
            "Instructions:\n"
            f"- Select exactly ONE best diagnosis from: {'/'.join(letters)}.\n"
            "- Give a concise justification (<=30 words).\n"
            '- Final line must be: "Final Answer: <LETTER>"\n'
        )
    return system, user

def load_diagnosisarena(split: str = "test", max_samples=None, seed: int = 42) -> pd.DataFrame:
    ds = load_dataset("shzyk/DiagnosisArena", split=split)
    records = []
    for i, ex in enumerate(ds):
        qid = str(ex.get("id") or i)
        ci = (ex.get("Case Information") or "").strip()
        pe = (ex.get("Physical Examination") or "").strip()
        dt = (ex.get("Diagnostic Tests") or "").strip()
        parts = []
        if ci: parts.append(f"Case Information: {ci}")
        if pe: parts.append(f"Physical Examination: {pe}")
        if dt: parts.append(f"Diagnostic Tests: {dt}")
        question = "\n".join(parts).strip() or ci

        opt_struct = ex.get("Options") or {}
        letters_sorted = [k for k in ["A", "B", "C", "D"] if k in opt_struct]
        options = [opt_struct[k] for k in letters_sorted]
        gold_letter = str(ex.get("Right Option", "")).strip().upper()
        gold_letter = re.sub(r"[^A-D]", "", gold_letter)

        if not qid or not options or not gold_letter:
            continue
        if gold_letter not in letters_sorted:
            continue

        records.append({
            "id": qid,
            "question": question,
            "options": options,
            "gold": gold_letter,
            "letters": letters_sorted,
        })
    df = pd.DataFrame(records)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df

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

def _print_models_containing(substr="gpt-5"):
    try:
        ms = client.models.list()
        ids = [m.id for m in ms.data]
        hits = [m for m in ids if substr in m]
        print("[DEBUG] Available models containing", substr, "=>", hits)
    except Exception as e:
        print("[DEBUG] list models failed:", e)

def call_gpt(system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> dict:
    start = time.time()
    try:
        if model.startswith("gpt-5") and not model.startswith("gpt-5-chat"):
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_output_tokens=1024,   
                reasoning={"effort": "low"},
            )
            text = _extract_text_from_responses(resp)
            usage = getattr(resp, "usage", None) or {}
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)

        else:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "timeout": TIMEOUT_S,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "response_format": {"type": "text"},
            }
            if not model.startswith("gpt-5-chat"):
                kwargs["temperature"] = 0

            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content if resp.choices else ""
            usage = getattr(resp, "usage", None) or {}
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        if not text:
            status = getattr(resp, "status", None)
            print(f"[WARN] Empty text from {model}. status={status} in={in_tok} out={out_tok}", flush=True)

        latency_ms = (time.time() - start) * 1000.0
        return {
            "ok": bool(text),
            "text": text or "",
            "latency_ms": latency_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "error": None if text else "empty_text"
        }
    except Exception as e:
        return {"ok": False, "text": "", "latency_ms": (time.time() - start) * 1000.0,
                "input_tokens": 0, "output_tokens": 0, "error": str(e)}

def eval_model_on_da(model: str, df: pd.DataFrame, kshot: int = 0, debug: bool = False) -> Tuple[pd.DataFrame, dict]:
    logs = []
    it = df.iterrows()
    total = len(df)
    if debug:
        total = min(5, total)
        it = list(df.sample(n=total, random_state=RANDOM_SEED).iterrows())

    for _, row in tqdm(
        it, total=total,
        desc=f"Evaluating {model} on DiagnosisArena {DA_SPLIT}{' [DEBUG]' if debug else ''} (kshot={kshot})"
    ):
        letters = tuple(row.get("letters", ("A", "B", "C", "D")))
        fewshot_block = build_fewshot_block(df, kshot=kshot, exclude_qid=row["id"])
        sys_p, usr_p = build_prompt_da(row["question"], row["options"], list(letters), model, fewshot_block)

        resp = call_gpt(sys_p, usr_p, model=model)

        if not resp.get("ok", False):
            print(f"[API-ERR] model={model} qid={row['id']} err={resp.get('error')}", flush=True)
            text = ""
            pred_choice = ""
            is_correct = 0
            refusal = 0
            err_missing = 1
            err_inconsistent = 0
            err_hallucinated = 0
            in_tok = resp.get("input_tokens", 0)
            out_tok = resp.get("output_tokens", 0)
            latency_ms = resp.get("latency_ms", 0.0)
        else:
            text = resp["text"] or ""
            in_tok = resp.get("input_tokens", 0)
            out_tok = resp.get("output_tokens", 0)
            latency_ms = resp.get("latency_ms", 0.0)

            pred_choice = extract_choice_strict(text, allowed_letters=letters)
            is_correct = 1 if (pred_choice == row["gold"]) else 0
            refusal = 1 if (not pred_choice and text and is_refusal(text)) else 0

            err_missing, err_inconsistent, err_hallucinated = 0, 0, 0
            if not is_correct and not refusal:
                err_missing, err_inconsistent, err_hallucinated = classify_errors(
                    pred_choice=pred_choice,
                    gold=row["gold"],
                    valid_letters=letters,
                    model_text=text,
                    is_correct=is_correct,
                )

        cost = estimate_cost_usd(model, in_tok, out_tok)
        logs.append(ItemLog(
            qid=row["id"], model=model, latency_ms=latency_ms,
            tokens_in=in_tok, tokens_out=out_tok, cost_usd=cost,
            gold=row["gold"], pred=pred_choice, is_correct=is_correct,
            refusal=refusal, err_missing=err_missing, err_inconsistent=err_inconsistent,
            err_hallucinated=err_hallucinated, raw_output=text
        ))

        if debug:
            print("\n[DEBUG SAMPLE]")
            print("QID:", row["id"])
            print("Gold:", row["gold"], " Pred:", pred_choice, " Correct:", bool(is_correct))
            print("Output:\n", (text[:1000] + ("..." if len(text) > 1000 else "")))

    df_log = pd.DataFrame([asdict(x) for x in logs])

    if df_log.empty:
        summary = {
            "model": model, "kshot": kshot, "n": 0,
            "accuracy": None, "avg_tokens_in": None, "avg_tokens_out": None,
            "total_cost_usd": 0.0, "avg_latency_ms": None,
            "missing_rate": None, "inconsistent_rate": None,
            "hallucinated_rate": None, "refusal_rate": None,
        }
        return df_log, summary

    summary = {
        "model": model,
        "kshot": kshot,
        "n": len(df_log),
        "accuracy": round(df_log["is_correct"].mean(), 4),
        "avg_tokens_in": round(df_log["tokens_in"].mean(), 2),
        "avg_tokens_out": round(df_log["tokens_out"].mean(), 2),
        "total_cost_usd": round(df_log["cost_usd"].sum(), 4),
        "avg_latency_ms": round(df_log["latency_ms"].mean(), 2),
        "missing_rate": round(df_log["err_missing"].mean(), 4),
        "inconsistent_rate": round(df_log["err_inconsistent"].mean(), 4),
        "hallucinated_rate": round(df_log["err_hallucinated"].mean(), 4),
        "refusal_rate": round(df_log["refusal"].mean(), 4),
    }
    return df_log, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=DA_SPLIT, help="HF split: test/dev/train")
    parser.add_argument("--models", type=str, default=",".join(MODELS),
                        help="Comma-separated models, e.g. 'gpt-5,gpt-5-chat-latest,gpt-4o'")
    parser.add_argument("--kshot", type=int, default=0, help="few-shot K (0 = zero-shot)")
    parser.add_argument("--debug", action="store_true", help="Sample mode with small N and raw dump.")
    parser.add_argument("--n", type=int, default=10, help="Sample size when --debug is on.")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Please set your key.")

    _print_models_containing("gpt-5")

    df = load_diagnosisarena(split=args.split)
    if args.debug:
        n = min(args.n, len(df))
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[DEBUG] Using {n} samples from DiagnosisArena/{args.split}")

    os.makedirs("results", exist_ok=True)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    summaries = []
    for model in model_list:
        df_log, summary = eval_model_on_da(model, df, kshot=args.kshot, debug=args.debug)
        suffix = "DEBUG" if args.debug else args.split
        out_csv = f"results/results_diagnosisarena_{suffix}_{model}_k{args.kshot}.csv"
        df_log.to_csv(out_csv, index=False)
        print(summary)
        summaries.append(summary)

    suffix = "DEBUG" if args.debug else args.split
    out_json = f"results/summary_diagnosisarena_{suffix}_{model}_k{args.kshot}.json"
    with open(out_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[INFO] Wrote summary to: {out_json}")

if __name__ == "__main__":
    main()