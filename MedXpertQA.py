import os, time, re, json, math, argparse, random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

from openai import OpenAI

OPENAI_API_KEY = "xxxxx"  # key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)



RANDOM_SEED = 42
random.seed(RANDOM_SEED)



MODELS = [
    "gpt-5",             
    "gpt-5-chat-latest",  
    "gpt-4o",             
]

PRICING_PER_1K = {
    "gpt-5": {"input": 0.005, "output": 0.015},
    "gpt-5-chat-latest": {"input": 0.005, "output": 0.015},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
}

TIMEOUT_S = 120

# Choose MedXpertQA subset/split
MEDX_SUBSET_DEFAULT = "Text"   # "Text" or "MM"
MEDX_SPLIT_DEFAULT  = "test"   # "dev" or "test"

# Output token limits
MAX_OUT_GPT5  = 1024     
MAX_OUT_CHAT  = 256     

REFUSAL_PATTERNS = [
    r"i\s+cannot\s+answer",
    r"i\s+can't\s+answer",
    r"unable\s+to\s+answer",
    r"i\s+refuse",
    r"i\s+will\s+not\s+provide",
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
    t = text.lower()
    return any(re.search(pat, t) for pat in REFUSAL_PATTERNS)

def extract_choice_strict(text: str, allowed_letters: Tuple[str, ...]) -> str:
    """Strictly extract one letter among allowed_letters (A–J)."""
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

def load_medxpertqa(subset: str = "Text", split: str = "test",
                    max_samples=None, seed: int = 42) -> pd.DataFrame:
    """
    Returns DataFrame:
      id, question, options(list[str]), gold(letter), letters(list[str]),
      medical_task, body_system, question_type, images
    """
    assert subset in ("Text", "MM")
    assert split in ("dev", "test")

    ds = load_dataset("TsinghuaC3I/MedXpertQA", name=subset, split=split)
    records = []
    for ex in ds:
        qid = str(ex.get("id", ""))
        question = (ex.get("question") or "").strip()
        options_dict = ex.get("options") or {}
        letters_sorted = sorted(options_dict.keys(), key=lambda x: x)  
        options = [options_dict[k] for k in letters_sorted]
        gold_letter = str(ex.get("label", "")).strip().upper()

        if not qid or not question or not options or not gold_letter:
            continue
        if gold_letter not in letters_sorted:
            continue

        records.append({
            "id": qid,
            "question": question,
            "options": options,
            "gold": gold_letter,
            "letters": letters_sorted,
            "medical_task": ex.get("medical_task"),
            "body_system": ex.get("body_system"),
            "question_type": ex.get("question_type"),
            "images": ex.get("images"),
        })

    df = pd.DataFrame(records)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df

def _format_example_row(r: Dict[str, Any]) -> str:
    L = r["letters"]
    q = (r["question"] or "").strip()[:350]
    opts = [o.strip()[:120] for o in r["options"][:len(L)]]
    labeled = [f"{L[i]}. {opts[i]}" for i in range(len(L))]
    return (
        "Example (read-only)\n"
        f"Question:\n{q}\n\n"
        "Options:\n" + "\n".join(labeled) + "\n"
        f"Final Answer: {r['gold']}\n"
        "----\n"
    )

def build_fewshot_for_item(pool_df: pd.DataFrame, target_row: dict, k: int, seed: int = 42) -> str:
    if k <= 0:
        return ""
    letters = "".join(target_row["letters"])  

    cand = pool_df[pool_df["letters"].apply(lambda L: "".join(L) == letters)]

    def _soft_mask(df):
        if df.empty:
            return df
        m = pd.Series([True] * len(df), index=df.index)
        for key in ["body_system", "question_type", "medical_task"]:
            val = target_row.get(key)
            if isinstance(val, str) and val.strip():
                m = m & (df[key] == val)
        return df[m] if m.any() else df

    cand2 = _soft_mask(cand)
    if cand2.empty:
        cand2 = cand
    if cand2.empty:
        return ""

    cand2 = cand2[cand2["id"] != target_row["id"]]
    if cand2.empty:
        return ""

    k = min(k, len(cand2))
    rows = cand2.sample(n=k, random_state=seed).to_dict(orient="records")

    return "Few-shot Examples:\n" + "".join(_format_example_row(r) for r in rows) + "\n"

def build_prompt_medx(
    question: str,
    options: List[str],
    letters: List[str],
    model_name: str = "",
    fewshot_block: str = ""
) -> Tuple[str, str]:
   
    labeled = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    header = (fewshot_block + "\n") if fewshot_block else ""

    system = (
        "You are answering medical MCQs for research evaluation. "
        "Follow the output format rules strictly."
    )
    user = (
        header +
        "TASK:\n"
        f"Question:\n{question.strip()}\n\n"
        "Options:\n" + "\n".join(labeled) + "\n\n"
        "Rules (CRITICAL):\n"
        f"- Allowed letters: {'/'.join(letters)}\n"
        "- Your FIRST LINE must be exactly: Final Answer: <LETTER>\n"
        "- If you add any text, put it AFTER the first line. Keep it under 30 words.\n"
    )
    return system, user

def _extract_text_from_responses(resp) -> str:
    """
    Universal extractor for Responses API (gpt-5) or mixed schema.
    Tries multiple fallbacks for robustness.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    if hasattr(resp, "output") and isinstance(resp.output, list):
        buf = []
        for o in resp.output:
            if isinstance(o, dict):
                if "content" in o and isinstance(o["content"], list):
                    for c in o["content"]:
                        if isinstance(c, dict) and "text" in c:
                            buf.append(c["text"])
                elif "text" in o:
                    buf.append(o["text"])
            elif hasattr(o, "content"):
                cont = o.content
                if isinstance(cont, list):
                    for c in cont:
                        t = getattr(c, "text", None)
                        if t:
                            buf.append(t)
        if buf:
            return "\n".join(buf).strip()

    if hasattr(resp, "choices"):
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    js = getattr(resp, "to_dict", lambda: {})()
    if isinstance(js, dict):
        for key in ["output_text", "output", "text"]:
            val = js.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        if "output" in js and isinstance(js["output"], list):
            for o in js["output"]:
                if isinstance(o, dict) and "content" in o:
                    if isinstance(o["content"], list):
                        for c in o["content"]:
                            if isinstance(c, dict) and "text" in c:
                                return c["text"].strip()

    return ""


def call_model(system_prompt: str, user_prompt: str, model: str) -> dict:
    start = time.time()
    try:
        if model == "gpt-5":
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_output_tokens=MAX_OUT_GPT5,
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
                "max_completion_tokens": MAX_OUT_CHAT,
                "response_format": {"type": "text"},
                "temperature": 0,
            }
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
            "error": None if text else "empty_text",
        }
    except Exception as e:
        latency_ms = (time.time() - start) * 1000.0
        print(f"[ERROR] call_model({model}) failed: {repr(e)}", flush=True)
        return {
            "ok": False,
            "text": "",
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        }

def eval_model_on_medxpert(
    model: str,
    df: pd.DataFrame,
    kshot: int = 0,
    shot_split: str = "dev",
    debug: bool = False
):
    logs = []
    it = df.iterrows()
    total = len(df)

    pool_df = load_medxpertqa(subset=MEDX_SUBSET_DEFAULT, split=shot_split)

    for _, row in tqdm(it, total=total,
                       desc=f"Evaluating {model} on MedXpertQA {MEDX_SUBSET_DEFAULT}/{MEDX_SPLIT_DEFAULT} (k={kshot})"):
        letters = tuple(row.get("letters", ()))
        fewshot_block = build_fewshot_for_item(pool_df, row, k=kshot, seed=RANDOM_SEED)

        sys_p, usr_p = build_prompt_medx(
            row["question"], row["options"], list(letters),
            model_name=model, fewshot_block=fewshot_block
        )

        resp = call_model(sys_p, usr_p, model=model)

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
            refusal = 1 if (not pred_choice and is_refusal(text)) else 0

            err_missing, err_inconsistent, err_hallucinated = 0, 0, 0
            if not is_correct and not refusal:
                err_missing, err_inconsistent, err_hallucinated = classify_errors(
                    pred_choice=pred_choice,
                    gold=row["gold"],
                    valid_letters=letters,
                    model_text=text,
                    is_correct=is_correct,
                )

        if debug:
            print(f"[DBG] {model} qid={row['id']} gold={row['gold']} pred={pred_choice} "
                  f"len(text)={len(text)} in={in_tok} out={out_tok}", flush=True)
            if text:
                first = text.splitlines()[0]
                print("      OUT:", first, flush=True)

        cost = estimate_cost_usd(model, in_tok, out_tok)
        logs.append(ItemLog(
            qid=row["id"], model=model, latency_ms=latency_ms,
            tokens_in=in_tok, tokens_out=out_tok, cost_usd=cost,
            gold=row["gold"], pred=pred_choice, is_correct=is_correct,
            refusal=refusal, err_missing=err_missing, err_inconsistent=err_inconsistent,
            err_hallucinated=err_hallucinated, raw_output=text
        ))

    df_log = pd.DataFrame([asdict(x) for x in logs])

    if df_log.empty:
        summary = {
            "model": model,
            "n": 0,
            "accuracy": None,
            "avg_tokens_in": None,
            "avg_tokens_out": None,
            "total_cost_usd": 0.0,
            "avg_latency_ms": None,
            "missing_rate": None,
            "inconsistent_rate": None,
            "hallucinated_rate": None,
            "refusal_rate": None,
        }
        return df_log, summary

    summary = {
        "model": model,
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
    parser.add_argument("--subset", default=MEDX_SUBSET_DEFAULT, choices=["Text", "MM"])
    parser.add_argument("--split",  default=MEDX_SPLIT_DEFAULT,  choices=["dev", "test"])
    parser.add_argument("--models", default="gpt-5,gpt-5-chat-latest,gpt-4o",
                        help="Comma-separated model IDs")
    parser.add_argument("--kshot", type=int, default=0, help="few-shot K (0=zero-shot)")
    parser.add_argument("--shot-split", default="dev", choices=["dev", "test"],
                        help="which split to draw few-shot examples from (pool)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n", type=int, default=10, help="sample size when --debug")
    args = parser.parse_args()

    df = load_medxpertqa(subset=args.subset, split=args.split, max_samples=args.max_samples)
    print(f"[INFO] Loaded MedXpertQA/{args.subset}/{args.split} with {len(df)} rows.")

    if args.debug:
        n = min(args.n, len(df))
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[DEBUG] Using {n} samples from MedXpertQA/{args.subset}/{args.split}")

    os.makedirs("results", exist_ok=True)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    summaries = []
    for model in model_list:
        print(f"[INFO] >>> Start evaluating model: {model} (kshot={args.kshot})", flush=True)
        df_log, summary = eval_model_on_medxpert(
            model=model, df=df, kshot=args.kshot, shot_split=args.shot_split, debug=args.debug
        )
        out_csv = f"results/results_medxpertqa_{args.subset.lower()}_{args.split}_{model}_k{args.kshot}.csv"
        df_log.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote per-item logs to: {out_csv}", flush=True)
        print("[RESULT]", summary, flush=True)
        summaries.append(summary)

    out_json = f"results/summary_medxpertqa_{args.subset.lower()}_{args.split}_k{args.kshot}.json"
    with open(out_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[INFO] Wrote summary to: {out_json}", flush=True)

if __name__ == "__main__":
    main()