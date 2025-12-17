#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate GPT-5 and GPT-4o on SLAKE closed-type Yes/No questions.

- Reads train.json (for few-shot pool) and test.json (for evaluation)
- Filters only closed questions with answers "Yes"/"No"
- Measures latency, token usage, and estimated total cost
- Outputs per-model CSV and summary JSON
"""

import os, re, json, time, base64, random, mimetypes
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ========== Configuration ==========
# Base directory can be overridden via SLAKE_DIR environment variable.
DATA_DIR = Path(os.getenv("SLAKE_DIR", "data/SLAKE"))
TRAIN_JSON = DATA_DIR / "train.json"
TEST_JSON = DATA_DIR / "test.json"
IMAGES_DIR = DATA_DIR / "imgs"

MODELS = ["gpt-5", "gpt-4o"]
K_LIST = [0, 1, 5]
# K_LIST = [0]
MAX_SAMPLES = None
TIMEOUT_S = 60
RANDOM_SEED = 42

OUTPUT_DIR = DATA_DIR / "results_slake_closed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICING_PER_1K = {
    "gpt-5": {"input": 0.005, "output": 0.015},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
}

REFUSAL_PATTERNS = [
    r"i\s+cannot\s+answer",
    r"i\s+can't\s+answer",
    r"unable\s+to\s+answer",
    r"as\s+an\s+ai",
    r"i\s+apologize",
    r"sorry[, ]\s+i\s+cannot",
]

# ========== OpenAI client ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ========== Utility functions ==========
def estimate_cost_usd(model, in_tokens, out_tokens):
    p = PRICING_PER_1K.get(model, {"input": 0, "output": 0})
    return (in_tokens / 1000) * p["input"] + (out_tokens / 1000) * p["output"]


def img_to_data_url(img_rel):
    path = IMAGES_DIR / img_rel
    if not path.exists():
        return None
    mime, _ = mimetypes.guess_type(path)
    if mime is None: mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def extract_final_answer(text):
    if not text: return ""
    m = re.search(r"final\s*answer\s*[:：]?\s*([A-Za-z0-9_ -]+)", text, flags=re.I)
    if m: return m.group(1).strip()
    return ""


def normalize_yesno(ans):
    s = str(ans).strip().lower()
    if s in ["yes", "y", "true", "1"]: return "Yes"
    if s in ["no", "n", "false", "0"]: return "No"
    return s.capitalize()  # e.g., “Liver”, “Spleen”


def is_refusal(text):
    if not text: return False
    t = text.lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def is_semantic_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    # 1) Normalize casing and whitespace
    p = pred.strip().lower()
    g = gold.strip().lower()

    # 2) Normalize MRI weighting abbreviations
    p = re.sub(r"\bt1w\b", "t1", p)
    p = re.sub(r"\bt2w\b", "t2", p)
    g = re.sub(r"\bt1w\b", "t1", g)
    g = re.sub(r"\bt2w\b", "t2", g)

    # 3) Remove common modifiers
    remove_words = ["weighted", "image", "scan", "mri", "ct", "x-ray", "xray",
                    "film", "photo", "picture", "slice", "view", "section"]
    for w in remove_words:
        p = re.sub(rf"\b{w}\b", "", p)
        g = re.sub(rf"\b{w}\b", "", g)
    p = p.strip()
    g = g.strip()

    # 4) Substring or set equality matches
    if p == g:
        return True
    if p in g or g in p:
        return True
    # Token set equality (e.g., "left lung" == "lung left")
    if set(p.split()) == set(g.split()):
        return True

    # 5) Article/plural handling
    p = re.sub(r"\b(the|a|an)\b", "", p)
    g = re.sub(r"\b(the|a|an)\b", "", g)
    if p.rstrip("s") == g.rstrip("s"):
        return True

    return False


def load_slake(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for ex in data:
        if str(ex.get("q_lang", "")).lower() != "en":
            continue
        if str(ex.get("answer_type", "")).upper() != "CLOSED":
            continue

        ans = normalize_yesno(ex.get("answer", ""))
        subtype = "yesno" if ans in ["Yes", "No"] else "eitheror"
        img_rel = str(ex.get("img_name", "")).strip()
        img_url = img_to_data_url(img_rel)
        if not img_url:
            continue

        out.append({
            "qid": ex.get("qid"),
            "img_name": img_rel,
            "question": str(ex.get("question", "")).strip(),
            "gold": ans,
            "subtype": subtype,
            "modality": ex.get("modality", ""),
            "content_type": ex.get("content_type", ""),
            "image_data_url": img_url
        })
    return out


def build_prompt(question, img_url, subtype="yesno", k_shot=0, examples=None):
    if subtype == "yesno":
        sys_p = ("You are a board-certified radiologist. "
                 "Answer Yes/No questions about medical images concisely. "
                 "Provide a short rationale, then 'Final Answer: Yes or No'.")
        answer_format = "Final Answer: Yes or No"
    else:
        sys_p = ("You are a board-certified radiologist. "
                 "Answer either-or questions about medical images concisely. "
                 "Provide a short rationale, then output 'Final Answer: <one of the given options>'.")
        answer_format = "Final Answer: one of the given options."

    parts = []
    if k_shot > 0 and examples:
        for ex in examples[:k_shot]:
            parts.append({"type": "text", "text": f"Example Question: {ex['question']}"})
            parts.append({"type": "image_url", "image_url": {"url": ex["image_data_url"]}})
            parts.append({"type": "text", "text": f"Final Answer: {ex['gold']}"})
    parts.append({"type": "text", "text": f"Question: {question}"})
    parts.append({"type": "image_url", "image_url": {"url": img_url}})
    parts.append({"type": "text", "text": f"Give a concise rationale, then output exactly:\n{answer_format}"})
    return sys_p, parts


def call_model(model, sys_p, parts):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it before calling the API.")
        
    start = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": parts}],
            timeout=TIMEOUT_S,
        )
        latency = (time.time() - start) * 1000
        txt = r.choices[0].message.content if r.choices else ""
        u = getattr(r, "usage", None) or {}
        return {"text": txt, "latency": latency,
                "in": getattr(u, "prompt_tokens", 0),
                "out": getattr(u, "completion_tokens", 0), "err": None}
    except Exception as e:
        return {"text": "", "latency": (time.time() - start) * 1000,
                "in": 0, "out": 0, "err": str(e)}


@dataclass
class Item:
    qid: Any;
    img_name: str;
    model: str;
    k_shot: int;
    subtype: str
    modality: str;
    content_type: str
    latency_ms: float;
    input_tokens: int;
    output_tokens: int;
    cost_usd: float
    gold: str;
    pred: str;
    is_correct: int;
    question: str;
    raw_output: str;
    error: Optional[str]


# ---------- Eval ----------
def eval_model(model, test_data, train_data, k_shot, max_samples=None):
    items = list(test_data)
    if max_samples and len(items) > max_samples:
        items = random.sample(items, max_samples)
    logs = []
    for ex in tqdm(items, desc=f"{model} k={k_shot}"):
        few = random.sample(train_data, min(k_shot, len(train_data))) if k_shot > 0 else []
        sys_p, parts = build_prompt(ex["question"], ex["image_data_url"],
                                    subtype=ex["subtype"], k_shot=k_shot, examples=few)
        r = call_model(model, sys_p, parts)
        pred = extract_final_answer(r["text"])
        corr = int(is_semantic_match(pred, ex["gold"]))
        cost = estimate_cost_usd(model, r["in"], r["out"])
        logs.append(Item(
            qid=ex["qid"], img_name=ex["img_name"], model=model, k_shot=k_shot,
            subtype=ex["subtype"], modality=ex["modality"], content_type=ex["content_type"],
            latency_ms=round(r["latency"], 2),
            input_tokens=r["in"], output_tokens=r["out"], cost_usd=round(cost, 6),
            gold=ex["gold"], pred=pred, is_correct=corr,
            question=ex["question"], raw_output=r["text"], error=r["err"]))
    return pd.DataFrame([asdict(x) for x in logs])


def results_analysis(data_folder: str | os.PathLike = "data/SLAKE", model: str = "gpt-5"):
    base_dir = Path(data_folder)
    csv_path = base_dir / "results_slake_closed" / f"results_slake_closed_{model}_k0.csv"
    output_json = base_dir / "results_slake_closed" / f"detailed_summary_slake_closed_{model}.json"

    df = pd.read_csv(csv_path)
    if "is_correct" not in df.columns:
        raise ValueError("❌ CSV must contain an 'is_correct' column to compute accuracy.")

    def get_summary(df, group_col):
        if group_col not in df.columns:
            return {}
        stats = (
            df.groupby(group_col)
            .agg(n_total=("is_correct", "size"),
                 n_correct=("is_correct", "sum"),
                 accuracy=("is_correct", "mean"))
            .reset_index()
            .sort_values("accuracy", ascending=False)
        )
        summary_dict = {
            row[group_col]: {
                "n_total": int(row["n_total"]),
                "n_correct": int(row["n_correct"]),
                "accuracy": round(row["accuracy"], 4)
            }
            for _, row in stats.iterrows()
        }
        return summary_dict

    # --- Category-level summaries ---
    subtype_summary = get_summary(df, "subtype")
    modality_summary = get_summary(df, "modality")
    content_summary = get_summary(df, "content_type")

    # --- Overall statistics ---
    overall_acc = df["is_correct"].mean()
    n_total = len(df)

    # --- Build JSON payload ---
    summary = {
        "dataset": "SLAKE (Closed-ended)",
        "model": "gpt-5",
        "n_total": int(n_total),
        "overall_accuracy": round(overall_acc, 4),
        "by_subtype": subtype_summary,
        "by_modality": modality_summary,
        "by_content_type": content_summary
    }

    # --- Save JSON ---
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[✅] Saved summary JSON to: {output_json}")
    print(f"Overall Accuracy: {overall_acc:.4f}\n")

    print("📊 Accuracy by subtype:")
    print(pd.DataFrame.from_dict(subtype_summary, orient="index"))
    print("\n📊 Accuracy by modality:")
    print(pd.DataFrame.from_dict(modality_summary, orient="index"))
    print("\n📊 Accuracy by content_type:")
    print(pd.DataFrame.from_dict(content_summary, orient="index"))


# ---------- Main ----------
def main():
    # random.seed(RANDOM_SEED)
    # print("[INFO] Loading SLAKE data...")
    # train = load_slake(TRAIN_JSON)
    # test = load_slake(TEST_JSON)
    # print(f"Train closed={len(train)} | Test closed={len(test)}")
    #
    # summaries = []
    # for model in MODELS:
    #     for k in K_LIST:
    #         df = eval_model(model, test, train, k, MAX_SAMPLES)
    #         csv_path = os.path.join(OUTPUT_DIR, f"results_slake_closed_{model}_k{k}.csv")
    #         df.to_csv(csv_path, index=False)
    #         print(f"[Saved] {csv_path}")
    #
    #         if len(df) == 0: continue
    #         s = {"model": model, "k_shot": k, "n": len(df),
    #              "accuracy": round(df.is_correct.mean(), 4),
    #              "avg_latency_ms": round(df.latency_ms.mean(), 2),
    #              "avg_tokens_in": round(df.input_tokens.mean(), 2),
    #              "avg_tokens_out": round(df.output_tokens.mean(), 2),
    #              "total_cost_usd": round(df.cost_usd.sum(), 4)}
    #         summaries.append(s)
    #         print(f"[{model} k={k}] acc={s['accuracy']} cost=${s['total_cost_usd']}")
    # with open(os.path.join(OUTPUT_DIR, "summary_slake_closed.json"), "w") as f:
    #     json.dump(summaries, f, indent=2)
    # print("Done.")

    results_analysis(model="gpt-4o")


if __name__ == "__main__":
    main()
