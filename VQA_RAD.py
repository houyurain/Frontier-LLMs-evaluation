"""
_*_CODING:UTF-8_*_
@Author: Yu Hou
@File: VQA_RAD.py
@Time: 9/28/25; 10:57 AM
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate GPT-5 and GPT-4o on VQA-RAD (Radiology VQA, multi-modal).
- Uses CloudFront images: https://d2rfm59k9u0hrr.cloudfront.net/medpix/img/full/{image_name}
- Closed-ended Yes/No subset by default (objective auto-eval).
- Supports 0/1/5-shot with multi-modal exemplars.
Outputs:
  - results_vqarad_<model>_<k_shot>.csv
  - summary_vqarad_k_shot.json
"""

import os, re, json, time, random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============== Config ==============
DATA_PATHS = [
    "VQA_RAD Dataset Public.json",
    os.path.join("/Users/hou00127/Google/Works/2025/Benchmarks/version_2/VQA_RAD", "VQA_RAD Dataset Public.json"),
]
CLOUDFRONT_PREFIX = "https://d2rfm59k9u0hrr.cloudfront.net/medpix/img/full/"

MODELS = [
    "gpt-5",
    "gpt-4o",
]

# Pricing (USD per 1K tokens) — 与你的 MedQA 脚本一致
PRICING_PER_1K = {
    "gpt-5": {"input": 0.005, "output": 0.015},  # $5/M in, $15/M out
    "gpt-4o": {"input": 0.0025, "output": 0.010},  # $2.5/M in, $10/M out
}

# 入口参数：设置 None 跑全量；或设置为整数做小样本快速测试
MAX_SAMPLES = None  # e.g., 50
RANDOM_SEED = 42
TIMEOUT_S = 60

# 只评估 Yes/No 子集（更易客观评估）
ONLY_YESNO_CLOSED = True

# （可选）是否评估 open-ended（需要二次调用 LLM 作为 Judge；默认关闭）
EVAL_OPEN_ENDED = False
JUDGE_MODEL = "gpt-5"

# Refusal 检测（与 MedQA 脚本同风格）
REFUSAL_PATTERNS = [
    r"i\s+cannot\s+answer",
    r"i\s+can't\s+answer",
    r"unable\s+to\s+answer",
    r"as\s+an\s+ai",
    r"i\s+apologize",
    r"sorry[, ]\s+i\s+cannot",
]

# ============== Client ==============
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


# ============== Utils ==============
def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens / 1000.0) * p["input"] + (out_tokens / 1000.0) * p["output"]


def is_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def read_vqarad(path_candidates: List[str]) -> List[Dict[str, Any]]:
    for p in path_candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Cannot find dataset in: {path_candidates}")


def normalize_yesno(s: str) -> str:
    if s is None:
        return ""
    x = s.strip().lower()
    if x in ("yes", "y", "true"):
        return "Yes"
    if x in ("no", "n", "false"):
        return "No"
    return ""


def make_image_url(image_name: str) -> str:
    return f"{CLOUDFRONT_PREFIX}{image_name}"


# 提取 Final Answer: Yes/No
def extract_final_answer(text: str) -> str:
    """
    提取 `Final Answer: ...` 的内容。
    - 支持 Yes/No (closed)
    - 支持 word/phrase (open)
    """
    if not text:
        return ""
    m = re.search(r"Final\s*Answer\s*[:：]?\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    return ""


# 与 MedQA 一致的错误分类框架（根据 Yes/No）
def classify_errors(pred_choice: str, gold: str, valid: Tuple[str, str], model_text: str, is_correct: int):
    """
    Returns (missing, inconsistent, hallucinated) as 0/1 (mutually exclusive).
    Priority: Missing > Invalid Label (as inconsistent) > Hallucinated > Inconsistent
    """
    if is_correct == 1:
        return 0, 0, 0
    # 1) Missing
    if not pred_choice:
        return 1, 0, 0
    # 2) Invalid -> inconsistent
    if pred_choice not in valid:
        return 0, 1, 0
    # 3) （保留位）Hallucinated：这里对 Yes/No 难定义，默认 0
    # 可按需扩展：例如检测互相矛盾的双重答案等
    return 0, 0, 0


@dataclass
class ItemLog:
    qid: str
    answer_type: str
    model: str
    k_shot: int
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


# ============== Data Loader (to DF like MedQA) ==============
def load_vqarad(max_samples: Optional[int] = None, seed: int = 42,
                only_yesno_closed: bool = True, include_open: bool = False) -> pd.DataFrame:
    random.seed(seed)
    data = read_vqarad(DATA_PATHS)

    rows = []
    for ex in data:
        qid = str(ex.get("qid"))
        q = str(ex.get("question") or "").strip()
        image_name = str(ex.get("image_name") or "").strip()
        answer_type = str(ex.get("answer_type") or "").strip().upper()
        gold_raw = str(ex.get("answer") or "").strip()

        if answer_type == "CLOSED":
            if only_yesno_closed:
                gold = normalize_yesno(gold_raw)
                if gold not in ("Yes", "No"):
                    continue
            else:
                gold = gold_raw
        elif answer_type == "OPEN":
            if not include_open:
                continue
            gold = gold_raw
        else:
            continue

        if not image_name or not q:
            continue

        rows.append({
            "id": qid,
            "question": q,
            "image_url": make_image_url(image_name),
            "gold": gold,
            "answer_type": answer_type,
            "image_name": image_name
        })

    df = pd.DataFrame(rows)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


# ============== Prompt (0/1/5-shot, multi-modal) ==============
def build_prompt_mm(question: str, k_shot: int, examples: List[Dict[str, Any]], answer_type: str = "CLOSED"):
    system_prompt = "You are a board-certified radiologist. Answer accurately based only on the image."
    parts: List[Dict[str, Any]] = []

    # few-shot examples (只用 CLOSED 做 few-shot 示例)
    if k_shot > 0 and examples:
        for ex in examples[:k_shot]:
            ex_q = ex["question"]
            ex_gold = ex["gold"]  # Yes/No
            ex_img = ex["image_url"]
            parts.append({"type": "text", "text": f"Example Question: {ex_q}"})
            parts.append({"type": "image_url", "image_url": {"url": ex_img}})
            parts.append({"type": "text", "text":
                "Answer briefly. Final line must be exactly:\n"
                "Final Answer: Yes\nor\nFinal Answer: No"
                          })
            parts.append({"type": "text", "text": f"Final Answer: {ex_gold}"})

    # main question
    if answer_type == "CLOSED":
        instruction = (
            f"Question: {question}\n\n"
            "First give a very brief reasoning (<= 30 words).\n"
            "Then on the final line, output strictly one of:\n"
            "Final Answer: Yes\n"
            "or\n"
            "Final Answer: No"
        )
    else:  # OPEN
        instruction = (
            f"Question: {question}\n\n"
            "First give a very brief reasoning (<= 30 words).\n"
            "Then on the final line, output strictly:\n"
            "Final Answer: <one word or short phrase>"
        )

    parts.append({"type": "text", "text": instruction})
    return system_prompt, parts


# ============== OpenAI call (multi-modal) ==============
def call_gpt_vision(system_prompt: str, user_parts: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_parts},
            ],
            timeout=TIMEOUT_S,
        )
        latency_ms = (time.time() - start) * 1000.0
        text = response.choices[0].message.content if response.choices else ""
        usage = getattr(response, "usage", None) or {}
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)
        return {
            "ok": True,
            "text": text or "",
            "latency_ms": latency_ms,
            "input_tokens": in_tok or 0,
            "output_tokens": out_tok or 0,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False, "text": "", "latency_ms": (time.time() - start) * 1000.0,
            "input_tokens": 0, "output_tokens": 0, "error": str(e),
        }


# ============== Evaluation ==============
def sample_examples(df: pd.DataFrame, k_shot: int, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    if k_shot <= 0 or df.empty:
        return []
    pool = df.sample(n=min(max(k_shot, 5), len(df)), random_state=seed)  # 多抽一点，保证充足
    return pool.to_dict(orient="records")


def eval_model_on_vqarad(model: str, df: pd.DataFrame, k_shot: int = 0):
    logs: List[ItemLog] = []

    # few-shot 示例池（只从 CLOSED 里抽，保证是 Yes/No 示例）
    examples = sample_examples(df[df["answer_type"] == "CLOSED"], k_shot=k_shot, seed=RANDOM_SEED)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model} (k={k_shot})"):
        sys_p, parts = build_prompt_mm(row["question"], k_shot=k_shot, examples=examples, answer_type=row["answer_type"])

        # 当前问题的图片
        parts.append({"type": "image_url", "image_url": {"url": row["image_url"]}})

        # 调用模型
        resp = call_gpt_vision(sys_p, parts, model=model)
        text = resp["text"]

        if row["answer_type"] == "CLOSED":
            pred_choice = extract_final_answer(text)
            is_correct = 1 if (pred_choice.capitalize() == row["gold"]) else 0
            refusal = 1 if is_refusal(text) else 0
            err_missing, err_inconsistent, err_hallucinated = classify_errors(
                pred_choice=pred_choice,
                gold=row["gold"],
                valid=("Yes", "No"),
                model_text=text,
                is_correct=is_correct,
            )

        else:  # OPEN
            pred_choice = extract_final_answer(text)
            refusal = 1 if is_refusal(pred_choice) else 0
            # 优先 exact match
            if pred_choice.lower() == row["gold"].lower():
                is_correct = 1
            else:
                # fallback: 用 GPT-5 Judge 判定语义等价
                try:
                    judge = judge_semantic_equivalence(row["gold"], pred_choice, model=JUDGE_MODEL)
                    is_correct = 1 if judge else 0
                except Exception as e:
                    is_correct = 0
            err_missing = err_inconsistent = err_hallucinated = 0

        cost = estimate_cost_usd(model, resp["input_tokens"], resp["output_tokens"])
        logs.append(ItemLog(
            qid=row["id"], answer_type=row["answer_type"], model=model, k_shot=k_shot,
            latency_ms=resp["latency_ms"],
            tokens_in=resp["input_tokens"], tokens_out=resp["output_tokens"],
            cost_usd=cost,
            gold=row["gold"], pred=pred_choice, is_correct=is_correct,
            refusal=refusal,
            err_missing=err_missing, err_inconsistent=err_inconsistent, err_hallucinated=err_hallucinated,
            raw_output=text,
        ))

    df_log = pd.DataFrame([asdict(x) for x in logs])

    if df_log.empty:
        summary = {
            "model": model,
            "k_shot": k_shot,
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
        "k_shot": k_shot,
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


# （可选）Open-ended 评估：用 GPT-5 作为 Judge（默认关闭）
def judge_semantic_equivalence(gold: str, pred: str, model: str = JUDGE_MODEL) -> bool:
    """
    让 LLM 作为评审，判断 pred 是否与 gold 语义等价（返回 True/False）。
    注意：会产生额外 API 成本；默认不启用。
    """
    sys_p = "You are an expert radiology QA judge."
    usr = (
        "Decide if the predicted answer is semantically equivalent to the gold answer.\n"
        "Answer strictly with one line: Final Answer: Yes or Final Answer: No\n\n"
        f"Gold: {gold}\n"
        f"Pred: {pred}\n"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}],
        timeout=TIMEOUT_S,
    )
    y = r.choices[0].message.content or ""
    lab = extract_final_answer(y)
    return lab == "Yes"


# ---------- Load JSON (robust for JSONL) ----------
def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------- Build mapping ----------
def build_qtype_map(json_path):
    data = load_json_or_jsonl(json_path)
    qtype_map = {}
    for item in data:
        qid = str(item.get("qid")).strip()
        qtype = item.get("question_type", "Other")
        qtype = ",".join([t.strip().upper() for t in qtype.split(",")]) if qtype else "OTHER"
        qtype_map[qid] = qtype
    print(f"[INFO] Loaded {len(qtype_map)} question_type mappings.")
    return qtype_map


def results_analysis(data_folder="/Users/hou00127/Google/Works/2025/Benchmarks/version_2/VQA_RAD/", model="gpt-5"):
    json_path = data_folder + "VQA_RAD Dataset Public.json"
    csv_path = data_folder + "results/results_vqarad_" + model + "_0.csv"
    output_csv = data_folder + "results/results_vqarad_with_type_" + model + ".csv"
    summary_json = data_folder + "results/summary_vqarad_by_type_" + model + ".json"

    # Load question type mapping
    qtype_map = build_qtype_map(json_path)

    # Load result CSV
    df = pd.read_csv(csv_path)
    id_col = next((c for c in df.columns if c.lower() in ["id", "qid", "question_id"]), None)
    if not id_col:
        raise ValueError("❌ CSV must contain an 'id' or 'qid' column.")

    # Normalize qid and add question_type
    df["qid_norm"] = df[id_col].astype(str).str.strip()
    df["question_type"] = df["qid_norm"].map(lambda x: qtype_map.get(x, "OTHER"))

    # 识别预测列与真实列
    pred_col = next((c for c in df.columns if c.lower() in ["pred", "prediction", "model_output", "response", "pred_answer"]), None)
    gold_col = next((c for c in df.columns if c.lower() in ["gold", "answer", "label", "correct_answer"]), None)
    if not pred_col or not gold_col:
        raise ValueError("❌ CSV must include columns for 'pred' and 'gold' answers.")

    # 计算是否正确（忽略大小写与空格）
    df["is_correct"] = df.apply(
        lambda r: str(r[pred_col]).strip().lower() == str(r[gold_col]).strip().lower(), axis=1
    )

    # 保存带 question_type 的 CSV
    df.to_csv(output_csv, index=False)
    print(f"[✅] Saved merged file with question_type → {output_csv}")

    # ---------- Accuracy summary ----------
    overall_acc = df["is_correct"].mean()
    type_stats = (
        df.groupby("question_type")
        .agg(n_total=("is_correct", "size"),
             n_correct=("is_correct", "sum"),
             accuracy=("is_correct", "mean"))
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    # ---------- Save summary JSON ----------
    summary = {
        "model": "gpt-5",
        "n_total": int(len(df)),
        "overall_accuracy": round(overall_acc, 4),
        "type_summary": {
            row["question_type"]: {
                "n_total": int(row["n_total"]),
                "n_correct": int(row["n_correct"]),
                "accuracy": round(row["accuracy"], 4)
            }
            for _, row in type_stats.iterrows()
        }
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[✅] Saved summary JSON → {summary_json}")
    print("\n📊 Per-type accuracy summary:")
    print(type_stats)


# ============== Main ==============
def main():
    # random.seed(RANDOM_SEED)
    # df = load_vqarad(max_samples=MAX_SAMPLES, seed=RANDOM_SEED, only_yesno_closed=ONLY_YESNO_CLOSED, include_open=EVAL_OPEN_ENDED)
    # summaries = []
    # for model in MODELS:
    #     for k_shot in [0, 1, 5]:
    #         df_log, summary = eval_model_on_vqarad(model, df, k_shot=k_shot)
    #         out_csv = f"results_vqarad_{model}_{k_shot}.csv"
    #         df_log.to_csv(out_csv, index=False)
    #         print(f"[Saved] {out_csv}")
    #         print(summary)
    #         summaries.append(summary)
    #
    # with open("summary_vqarad_k_shot.json", "w", encoding="utf-8") as f:
    #     json.dump(summaries, f, indent=2)
    # print("[Saved] summary_vqarad_k_shot.json")

    results_analysis(model="gpt-5")


if __name__ == "__main__":
    main()
