"""Evaluation utilities for the PMC-VQA benchmark."""
import os, re, json, time, random, base64, mimetypes
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import shutil


def generate_image_subset(source_dir: Path, destination_dir: Path, csv_path: Path) -> None:
    """Copy only the images referenced in the CSV (plus few-shot examples) to a smaller folder."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    figure_files = set(df["Figure_path"].astype(str).tolist())

    extra_examples = {
        "PMC1064097_F1.jpg",
        "PMC1064097_F2.jpg",
        "PMC1064097_F4.jpg",
        "PMC1064097_F6.jpg",
        "PMC1064097_F7.jpg",
    }

    all_needed = figure_files.union(extra_examples)

    print(f"Total images to copy: {len(all_needed)}")

    missing = []
    for img_name in all_needed:
        src_path = source_dir / img_name
        dst_path = destination_dir / img_name
        if src_path.exists():
            shutil.copy(src_path, dst_path)
        else:
            missing.append(img_name)

    print(f"✅ Finished copying. Copied {len(all_needed) - len(missing)} images.")
    if missing:
        print("⚠️ Missing images:", missing)


"""
Evaluate GPT-5 and GPT-4o on PMC-VQA (multiple-choice A/B/C/D).
- Reads test_clean.csv
- Loads images from a local folder (base64 -> image_url data URI)
- Supports 0/1/5-shot (few-shot examples defined in build_prompt_mc)
- Handles Answer_label anomalies
"""

# ========= Paths =========
DATA_DIR = Path(os.getenv("PMC_VQA_DIR", Path("data/PMC_VQA")))
TEST_CSV = DATA_DIR / "test_clean.csv"
IMAGES_DIR = DATA_DIR / "image_subset"


# ========= Eval Config =========
MODELS = ["gpt-5", "gpt-4o"]
K_LIST = [0, 1, 5]  # Default runs 0-shot; extend to [0, 1, 5] as needed
MAX_SAMPLES = None  # e.g., 100 for smoke test; None = full set
RANDOM_SEED = 42
TIMEOUT_S = 90

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

# ========= OpenAI Client =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ========= Utilities =========
def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens / 1000.0) * p["input"] + (out_tokens / 1000.0) * p["output"]


def is_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def img_to_data_url(img_path: str) -> Optional[str]:
    if not os.path.exists(img_path):
        return None
    mime, _ = mimetypes.guess_type(img_path)
    if mime is None:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        else:
            mime = "application/octet-stream"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def norm_text(x: str) -> str:
    if x is None: return ""
    s = str(x).strip()
    s = re.sub(r"^[A-Da-d]\s*[:：]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def resolve_answer_label(row: Dict[str, Any]) -> str:
    """
    Return A/B/C/D; handle Answer_label anomalies like stray quotes.
    """
    label_raw = str(row.get("Answer_label", "")).strip().upper()
    if label_raw in ["A", "B", "C", "D"]:
        return label_raw
    ans = norm_text(row.get("Answer", ""))
    choices = {
        "A": norm_text(row.get("Choice A", "")),
        "B": norm_text(row.get("Choice B", "")),
        "C": norm_text(row.get("Choice C", "")),
        "D": norm_text(row.get("Choice D", "")),
    }
    for k, v in choices.items():
        if ans and v and ans == v: return k
    for k, v in choices.items():
        if ans and v and (ans in v or v in ans): return k
    return ""


def extract_final_choice(text: str) -> str:
    if not text: return ""
    m = re.search(r"final\s*answer\s*[:：]?\s*([ABCD])\b", text, flags=re.I)
    if m: return m.group(1).upper()
    return ""


# ========= Prompt =========
def render_choices(a, b, c, d) -> str:
    return f"A) {a}\nB) {b}\nC) {c}\nD) {d}"


def build_prompt_mc(question: str, a: str, b: str, c: str, d: str, k_shot: int = 0):
    """
    Build a multimodal MCQ prompt.
    Few-shot examples (6 images) are embedded below.
    """
    system_prompt = (
        "You are a board-certified clinician. Read the medical figure and answer the MCQ accurately."
    )
    parts: List[Dict[str, Any]] = []

    # ===== Few-shot templates (6 images) =====
    if k_shot > 0:
        few_shot_examples = [
            {
                "figure": "PMC1064097_F1.jpg",
                "question": "Example Q1: What is the uptake pattern in the breast? ",
                "choices": {"A": " Diffuse uptake pattern", "B": "Focal uptake pattern", "C": " No uptake pattern", "D": "Cannot determine from the information given"},
                "final_answer": "B"
            },
            {
                "figure": "PMC1064097_F2.jpg",
                "question": "Example Q2: What radiological technique was used to confirm the diagnosis?",
                "choices": {"A": " Mammography", "B": "CT Scan", "C": "MRI", "D": "X-ray"},
                "final_answer": "A"
            },
            {
                "figure": "PMC1064097_F4.jpg",
                "question": "Example Q3: Where were the microcalcifications located in the mammography image?",
                "choices": {"A": "Behind the nipple", "B": "Above the nipple", "C": "Below the nipple", "D": " Around the nipple"},
                "final_answer": "A"
            },
            {
                "figure": "PMC1064097_F6.jpg",
                "question": "Example Q4: What is the name of the radiopharmaceutical used for scintimammography in the upper row of the image?",
                "choices": {"A": "99mTc-MIBI", "B": "99mTc-(V)DMSA", "C": "Fludeoxyglucose", "D": "F-18 Sodium Fluoride"},
                "final_answer": "B"
            },
            {
                "figure": "PMC1064097_F6.jpg",
                "question": "Example Q5: What is the name of the radiopharmaceutical used for scintimammography in the bottom row of the image?",
                "choices": {"A": "99mTc-MIBI", "B": "99mTc-(V)DMSA", "C": "F-18 Sodium Fluoride", "D": "Fludeoxyglucose"},
                "final_answer": "A"
            },
            {
                "figure": "PMC1064097_F7.jpg",
                "question": "Example Q6: Which of the following imaging techniques have been used in the study of the patient?",
                "choices": {"A": "CT scan", "B": "X-ray", "C": "MRI", "D": "99mTc-(V)DMSA autoradiogram"},
                "final_answer": "C"
            },
        ]

        for ex in few_shot_examples[:k_shot]:
            img_path = os.path.join(IMAGES_DIR, ex["figure"])
            data_url = img_to_data_url(img_path)
            if not data_url:
                continue
            parts.append({"type": "text", "text": f"Question: {ex['question']}"})
            parts.append({"type": "text", "text": render_choices(
                ex['choices']['A'], ex['choices']['B'], ex['choices']['C'], ex['choices']['D']
            )})
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
            parts.append({"type": "text", "text":
                "Give a very brief rationale (<= 30 words).\n"
                "Then output exactly one line:\n"
                "Final Answer: A\nor B\nor C\nor D"
                          })
            parts.append({"type": "text", "text": f"Final Answer: {ex['final_answer']}"})

    # ===== Main question =====
    parts.append({"type": "text", "text": f"Question: {question}"})
    parts.append({"type": "text", "text": render_choices(a, b, c, d)})
    parts.append({"type": "text", "text":
        "Give a very brief rationale (<= 30 words).\n"
        "Then output exactly one line:\n"
        "Final Answer: A\nor B\nor C\nor D"
                  })

    return system_prompt, parts


# ========= OpenAI call =========
def call_model(system_prompt: str, user_parts: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it before calling the API.")
        
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_parts},
            ],
            timeout=TIMEOUT_S,
        )
        latency_ms = (time.time() - start) * 1000.0
        text = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None) or {}
        return {
            "ok": True, "text": text or "", "latency_ms": latency_ms,
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "error": None
        }
    except Exception as e:
        return {"ok": False, "text": "", "latency_ms": (time.time() - start) * 1000.0,
                "input_tokens": 0, "output_tokens": 0, "error": str(e)}


# ========= Data =========
def load_test_dataframe() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    rows = []
    for i, row in df.iterrows():
        label = resolve_answer_label(row)
        if label not in ("A", "B", "C", "D"): continue
        fig_name = os.path.basename(str(row["Figure_path"]).strip())
        img_path = os.path.join(IMAGES_DIR, fig_name)
        data_url = img_to_data_url(img_path)
        if not data_url: continue
        rows.append({
            "id": f"{fig_name}__{i}",
            "figure_path": fig_name,
            "question": str(row["Question"]).strip(),
            "A": str(row["Choice A"]).strip(),
            "B": str(row["Choice B"]).strip(),
            "C": str(row["Choice C"]).strip(),
            "D": str(row["Choice D"]).strip(),
            "gold_label": label,
            "image_data_url": data_url
        })
    df2 = pd.DataFrame(rows)
    if MAX_SAMPLES and len(df2) > MAX_SAMPLES:
        df2 = df2.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    return df2


# ========= Logging =========
@dataclass
class ItemLog:
    qid: str
    figure_path: str
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
    raw_output: str


def eval_model_on_df(model: str, df: pd.DataFrame, k_shot: int):
    logs: List[ItemLog] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model}-k={k_shot}"):
        sys_p, parts = build_prompt_mc(row["question"], row["A"], row["B"], row["C"], row["D"], k_shot=k_shot)
        parts.append({"type": "image_url", "image_url": {"url": row["image_data_url"]}})
        resp = call_model(sys_p, parts, model=model)
        text = resp["text"]
        pred_choice = extract_final_choice(text)
        is_correct = 1 if pred_choice == row["gold_label"] else 0
        refusal = 1 if is_refusal(text) else 0
        cost = estimate_cost_usd(model, resp["input_tokens"], resp["output_tokens"])
        logs.append(ItemLog(
            qid=row["id"], figure_path=row["figure_path"],
            model=model, k_shot=k_shot,
            latency_ms=resp["latency_ms"], tokens_in=resp["input_tokens"], tokens_out=resp["output_tokens"],
            cost_usd=cost, gold=row["gold_label"], pred=pred_choice,
            is_correct=is_correct, refusal=refusal, raw_output=text
        ))
    df_log = pd.DataFrame([asdict(x) for x in logs])
    summary = {
        "model": model, "k_shot": k_shot, "n": len(df_log),
        "accuracy": round(df_log["is_correct"].mean(), 4) if len(df_log) > 0 else None,
        "avg_tokens_in": df_log["tokens_in"].mean() if len(df_log) > 0 else None,
        "avg_tokens_out": df_log["tokens_out"].mean() if len(df_log) > 0 else None,
        "total_cost_usd": df_log["cost_usd"].sum() if len(df_log) > 0 else 0.0,
        "avg_latency_ms": df_log["latency_ms"].mean() if len(df_log) > 0 else None,
        "refusal_rate": df_log["refusal"].mean() if len(df_log) > 0 else None
    }
    return df_log, summary


# ========= Main =========
def main():
    import argparse

    parser = argparse.ArgumentParser(description="PMC-VQA evaluation utilities.")
    parser.add_argument("--export-metadata", type=Path, help="Optional path to export flattened test metadata CSV.")
    parser.add_argument("--prepare-subset", action="store_true",
                        help="Copy only the referenced images into the image_subset folder.")
    parser.add_argument("--source-images", type=Path, default=DATA_DIR / "images",
                        help="Folder containing the full set of PMC-VQA images.")
    args = parser.parse_args()
    
    random.seed(RANDOM_SEED)
    df_test = load_test_dataframe()
    print(f"[INFO] Loaded {len(df_test)} test rows from {TEST_CSV}.")

    if args.prepare_subset:
        generate_image_subset(args.source_images, IMAGES_DIR, TEST_CSV)

    if args.export_metadata:
        df_test[["id", "figure_path", "question", "A", "B", "C", "D", "gold_label"]].to_csv(
            args.export_metadata, index=False
        )
        print(f"[Saved] Exported metadata to {args.export_metadata}")

    # Example evaluation loop (disabled by default to avoid unintended API usage):
    # summaries = []
    # for model in MODELS:
    #     for k in K_LIST:
    #         df_log, summary = eval_model_on_df(model, df_test, k)
    #         out_csv = f"results_pmc_vqa_{model}_{k}.csv"
    #         df_log.to_csv(out_csv, index=False)
    #         print(f"[Saved] {out_csv}")
    #         print(summary)
    #         summaries.append(summary)
    # with open("summary_pmc_vqa_k_shot.json", "w", encoding="utf-8") as f:
    #     json.dump(summaries, f, indent=2)
    # print("[Saved] summary_pmc_vqa_k_shot.json")


if __name__ == '__main__':
    main()
