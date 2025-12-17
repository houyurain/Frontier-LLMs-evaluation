"""
_*_CODING:UTF-8_*_
@Author: Yu Hou
@File: MedMCQA.py
@Time: 9/26/25; 10:05 AM
"""

import os, time, re, json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, asdict
from openai import OpenAI

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# Models to evaluate
MODELS = [
    "gpt-5",
    "gpt-4o",
]

# Pricing (USD per 1K tokens, official OpenAI 2025 pricing)
PRICING_PER_1K = {
    "gpt-5": {"input": 0.005, "output": 0.015},  # $5/M in, $15/M out
    "gpt-4o": {"input": 0.0025, "output": 0.010},  # $2.5/M in, $10/M out
}

MAX_SAMPLES = 10  # set None for full test
RANDOM_SEED = 42
TIMEOUT_S = 60

# Refusal detection patterns
REFUSAL_PATTERNS = [
    r"\bi('?m| am)\s(sorry|unable|not able)\b",
    r"\bcan(not|'t)\b\s(answer|comply|help)",
    r"\bas an ai\b",
]


# ---------- Data structure ----------
@dataclass
class ItemLog:
    qid: str
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


# ---------- Utils ----------
def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens / 1000.0) * p["input"] + (out_tokens / 1000.0) * p["output"]


def is_refusal(text: str) -> bool:
    return any(re.search(pat, text.lower()) for pat in REFUSAL_PATTERNS)


FINAL_ANSWER_REGEX = re.compile(
    r"(?:final\s*answer|correct\s*answer|answer\s*(?:is|:))\s*\(?\s*([A-D])\s*\)?",
    flags=re.I
)


def extract_choice(text: str) -> str:
    """从 Final Answer 行提取 A/B/C/D，如果没有就返回空"""
    m = FINAL_ANSWER_REGEX.search(text or "")
    return m.group(1).upper() if m else ""


def has_conflicting_final_answers(text: str, pred: str, allowed_labels: set) -> bool:
    """检查是否有多个相互矛盾的 Final Answer"""
    letters = {m.group(1).upper() for m in FINAL_ANSWER_REGEX.finditer(text or "")}
    contradictions = {l for l in letters if l in allowed_labels and l != pred}
    return len(contradictions) > 0


def build_prompt_medmcqa(question: str, options: list[str], k_shot: int = 0) -> tuple[str, str]:
    letters = "ABCD"
    opts = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])
    system = (
        "You are a board-certified physician. Answer accurately. "
        "Do not reveal chain-of-thought."
    )

    examples = ""
    if k_shot == 1:
        examples += (
            "Example question:Normal visual acuity of 6/6 is achieved by the age of:\n"
            "A. 2 years\nB. 3 years\nC. 5 years\nD. 7 years"
            "Final Answer: B"
        )

    if k_shot == 5:
        examples += (
            "Example question:Primary feature of small intestinal obstructionf:\n"
            "A. Fever\nB. High peristalsis with colic\nC. Abdominal distension\nD. Empty rectum"
            "Final Answer: B"
            "Example question:Regarding unicornuate uterus true is:\n"
            "A. Defect in fusion of mullerian ducts\nB. Single uterus with 2 fallopian tubes\nC. One duct is completely absent\nD. Best reproductive outcome"
            "Final Answer: C"
            "Example question:Characteristic lab findings of hemophilia A are –a) ↑ PTb) ↑ aPTTc) X–linked recessived) Presence of 30% of factor level express the diseasee) Increased bleeding time:\n"
            "A. a\nB. c\nC. ac\nD. bc"
            "Final Answer: D"
            "Example question:What is true about HER2/neu overexpression in Ca breast:\n"
            "A. Good prognosis\nB. Responds well to taxanes\nC. Responds well to monoclonal antibodies\nD. Seen only in breast cancer"
            "Final Answer: C"
            "Example question:Which one of the following is a muscle splitting incision: September 2012:\n"
            "A. Kocher's\nB. Rutherford-Morris\nC. Pfannensteil\nD. Lanz"
            "Final Answer: D"
        )
    user = (
        "This is a single-answer medical MCQ. "
        "Respond with a brief rationale (<= 30 words), then output the final choice.\n"
        "Final line must be exactly: Final Answer: <LETTER>\n"
        "Allowed letters: A, B, C, D"
        f"{examples}\n"
        f"{question}\n\n{opts}\n\n"
    )
    return system, user


# ---------- Data loading ----------
VALID_LETTERS = ("A", "B", "C", "D")


def _clean(s):
    return (s or "").strip()


def _map_cop_value_to_letter(v, num_opts=4):
    """
    v: 单个 cop 值（int/str）
    优先按 1 基（1->A，…），不在范围再按 0 基（0->A，…）
    """
    if v is None:
        return None
    if isinstance(v, str):
        vs = v.strip().upper()
        if vs in VALID_LETTERS:
            return vs
        if vs.isdigit():
            v = int(vs)
        else:
            return None
    try:
        vi = int(v)
    except Exception:
        return None

    # 先按 1 基
    if 1 <= vi <= num_opts:
        return "ABCD"[vi - 1]
    # 再按 0 基
    if 0 <= vi < num_opts:
        return "ABCD"[vi]
    return None


def load_medmcqa(split="train", max_samples=None, seed=42) -> pd.DataFrame:
    """
    适配 lighteval/med_mcqa（字段：question, exp, cop, opa, opb, opc, opd, subject_name, topic_name, id, choice_type）
    注意：该镜像实测 gold 等价为“单选”。统一视为 single。
    返回列：id, question, options(list[str]), gold(str), qtype("single"), meta(dict)
    """
    ds = load_dataset("lighteval/med_mcqa", split=split)
    records = []
    for ex in ds:
        qid = str(ex.get("id", ""))
        question = _clean(ex.get("question", ""))

        # 选项（A/B/C/D 顺序）
        options = [ex.get("opa"), ex.get("opb"), ex.get("opc"), ex.get("opd")]
        options = [_clean(o) for o in options if isinstance(o, str)]
        options = options[:4]
        if len(options) < 2 or not question:
            continue

        # 正确答案（cop 可能为 -1 / 0-基 / 1-基）
        cop = ex.get("cop", None)
        gold_letter = _map_cop_value_to_letter(cop, num_opts=len(options))
        if gold_letter not in VALID_LETTERS:
            continue  # 无效/缺失答案的样本跳过

        meta = {
            "subject": _clean(ex.get("subject_name")),
            "topic": _clean(ex.get("topic_name")),
        }

        records.append({
            "id": qid if qid else f"medmcqa_{len(records)}",
            "question": question,  # 不拼 exp，避免泄露
            "options": options,
            "gold": gold_letter,  # 这里就是单个字母
            "qtype": "single",  # 强制单选
            "meta": meta
        })

    df = pd.DataFrame(records)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


def load_medmcqa_stratified(split="train", samples_per_subject=200, seed=42) -> pd.DataFrame:
    ds = load_dataset("lighteval/med_mcqa", split=split)
    records = []
    for ex in ds:
        qid = str(ex.get("id", ""))
        question = (ex.get("question") or "").strip()

        # 选项（A/B/C/D 顺序）
        options = [ex.get("opa"), ex.get("opb"), ex.get("opc"), ex.get("opd")]
        options = [_clean(o) for o in options if isinstance(o, str)]
        options = options[:4]
        if len(options) < 2 or not question:
            continue

        # 正确答案（cop 可能为 -1 / 0-基 / 1-基）
        cop = ex.get("cop", None)
        gold_letter = _map_cop_value_to_letter(cop, num_opts=len(options))
        if gold_letter not in VALID_LETTERS:
            continue  # 无效/缺失答案的样本跳过

        subject = (ex.get("subject_name") or "").strip()
        topic = (ex.get("topic_name") or "").strip()

        records.append({
            "id": qid if qid else f"medmcqa_{len(records)}",
            "question": question,
            "options": options,
            "gold": gold_letter,
            "qtype": "single",
            "subject": subject,
            "topic": topic,
        })

    df = pd.DataFrame(records)

    # ✅ 分层采样按 subject
    df_sampled = (
        df.groupby("subject")
        .apply(lambda g: g.sample(n=min(len(g), samples_per_subject), random_state=seed))
        .reset_index(drop=True)
    )

    return df_sampled


# ---------- OpenAI call ----------
def call_gpt(system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> dict:
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = (time.time() - start) * 1000.0
        text = response.choices[0].message.content if response.choices else ""
        usage = getattr(response, "usage", None) or {}
        in_tok = getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "completion_tokens", 0)
        return {
            "ok": True,
            "text": text,
            "latency_ms": latency_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "error": None,
        }
    except Exception as e:
        print(e)
        return {"ok": False, "text": "", "latency_ms": (time.time() - start) * 1000.0,
                "input_tokens": 0, "output_tokens": 0, "error": str(e)}


# ---------- Evaluation ----------
def eval_model_on_medqa(model: str, df: pd.DataFrame, k_shot: int = 0):
    logs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model}"):
        sys_p, usr_p = build_prompt_medmcqa(row["question"], row["options"], k_shot)
        resp = call_gpt(sys_p, usr_p, model=model)
        text = resp["text"]
        pred = extract_choice(text)
        gold = row.gold

        # 初始化
        is_correct = 0
        refusal = 0
        err_missing = 0
        err_inconsistent = 0
        err_hallucinated = 0

        # Step 1: 判断是否答对
        if pred and pred == gold:
            is_correct = 1
        else:
            # Step 2: 判拒答
            if is_refusal(text):
                refusal = 1
            else:
                # Step 3: 判错误类型
                if not pred:
                    err_missing = 1
                elif pred not in {"A", "B", "C", "D"}:
                    err_inconsistent = 1
                elif pred != gold:
                    # 合法但答错
                    err_inconsistent = 1
                    # 检查是否有多个矛盾 Final Answer
                    if has_conflicting_final_answers(text, pred, {"A", "B", "C", "D"}):
                        err_hallucinated = 1

        cost = estimate_cost_usd(model, resp["input_tokens"], resp["output_tokens"])
        logs.append(ItemLog(
            qid=row["id"], model=model, k_shot=k_shot, latency_ms=resp["latency_ms"],
            tokens_in=resp["input_tokens"], tokens_out=resp["output_tokens"], cost_usd=cost,
            gold=row["gold"], pred=pred, is_correct=is_correct,
            refusal=refusal, err_missing=err_missing, err_inconsistent=err_inconsistent,
            err_hallucinated=err_hallucinated, raw_output=text
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


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]


def results_analysis(data_folder="/Users/hou00127/Google/Works/2025/Benchmarks/version_2/MedMCQA/", model="gpt-5"):
    # ---------- Paths ----------
    test_json_path = data_folder + "train.json"  # 原始测试集
    result_csv_path = data_folder + "results_medmcqa_" + model + ".csv"  # 模型结果
    merged_csv_path = data_folder + "results_medmcqa_" + model + "_merged.csv"
    summary_json_path = data_folder + "subject_summary_medmcqa_" + model + ".json"

    # ---------- Load test metadata ----------
    test_data = load_json_or_jsonl(test_json_path)

    # 构建 id → metadata 映射
    meta_map = {}
    for item in test_data:
        qid = item.get("id")
        subj = item.get("subject_name")
        topic = item.get("topic_name")
        meta_map[qid] = {"subject_name": subj, "topic_name": topic}

    print(f"[INFO] Loaded {len(meta_map)} entries from test.json")

    # ---------- Load model results ----------
    df = pd.read_csv(result_csv_path)

    # 自动检测 id 列名
    id_col = None
    for c in df.columns:
        if c.lower() in ["id", "qid", "question_id"]:
            id_col = c
            break
    if id_col is None:
        raise ValueError("❌ No column named 'id' or 'qid' found in result CSV!")

    # ---------- Merge ----------
    df["subject_name"] = df[id_col].map(lambda x: meta_map.get(x, {}).get("subject_name", "Unknown"))
    df["topic_name"] = df[id_col].map(lambda x: meta_map.get(x, {}).get("topic_name", "Unknown"))

    # ---------- Compute Accuracy ----------
    # 自动检测预测列和真实列（常见列名：pred, prediction, model_output, gold, answer, label）
    pred_col = None
    gold_col = None
    for c in df.columns:
        if c.lower() in ["pred", "prediction", "model_output", "response", "pred_answer"]:
            pred_col = c
        if c.lower() in ["gold", "answer", "label", "correct_answer"]:
            gold_col = c
    if pred_col is None or gold_col is None:
        raise ValueError("❌ Please make sure your CSV has 'pred' and 'gold' columns!")

    # 去除空格、统一大小写再比较
    df["is_correct"] = df.apply(
        lambda r: str(r[pred_col]).strip().lower() == str(r[gold_col]).strip().lower(), axis=1
    )

    # 整体准确率
    overall_acc = df["is_correct"].mean()

    # 按 subject_name 分组准确率
    subject_stats = (
        df.groupby("subject_name")
        .agg(n_total=("is_correct", "size"),
             n_correct=("is_correct", "sum"),
             accuracy=("is_correct", "mean"))
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    # ---------- Save results ----------
    df.to_csv(merged_csv_path, index=False)
    print(f"[✅ Saved] Merged results to: {merged_csv_path}")

    summary = {
        "model": "gpt-5",
        "n_total": len(df),
        "overall_accuracy": round(overall_acc, 4),
        "subject_summary": {
            row["subject_name"]: {
                "n_total": int(row["n_total"]),
                "n_correct": int(row["n_correct"]),
                "accuracy": round(row["accuracy"], 4)
            }
            for _, row in subject_stats.iterrows()
        }
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[✅ Saved] Summary JSON to: {summary_json_path}")
    print(f"\n📊 Overall Accuracy: {overall_acc:.4f}")
    print("📘 Accuracy by subject:")
    print(subject_stats)


def main():
    # df = load_medmcqa(max_samples=MAX_SAMPLES)
    # for idx in range(1, 6):
    #     print(df.loc[idx, "question"].strip())
    #     letters = "ABCD"
    #     options = df.loc[idx, "options"]
    #     print("\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)]))
    #     print(df.loc[idx, "gold"])

    # df = load_medmcqa_stratified(split="train", samples_per_subject=200)
    # print(len(df), "samples")
    # print(df["subject"].value_counts())
    #
    # summaries = []
    # for k_shot in [1, 5]:
    #     for model in MODELS:
    #         df_log, summary = eval_model_on_medqa(model, df, k_shot)
    #         df_log.to_csv(f"results_medmcqa_{model}_{k_shot}.csv", index=False)
    #         print(summary)
    #         summaries.append(summary)
    # with open("summary_medmcqa_k_shot.json", "w") as f:
    #     json.dump(summaries, f, indent=2)

    results_analysis(model="gpt-4o")


if __name__ == "__main__":
    main()
