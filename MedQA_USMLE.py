"""
_*_CODING:UTF-8_*_
@Author: Yu Hou
@File: MedQA_USMLE.py
@Time: 9/24/25; 10:44 AM
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate GPT-5 and GPT-4o on MedQA (USMLE).
Assumption: always 4 options (A/B/C/D).
For Hugging Face BigBio version (requires datasets<3.0).
Outputs:
  - results_medqa_<model>.csv
  - summary_medqa.json
"""

import os, time, re, json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Tuple
from openai import OpenAI

# ---------- CONFIG ----------
# !!! Put your API key here !!!
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

MAX_SAMPLES = None  # set None for full test
RANDOM_SEED = 42
TIMEOUT_S = 60

# Refusal detection patterns
REFUSAL_PATTERNS = [
    r"i\s+cannot\s+answer",
    r"i\s+can't\s+answer",
    r"unable\s+to\s+answer",
    r"as\s+an\s+ai",
    r"i\s+apologize",
    r"sorry[, ]\s+i\s+cannot",
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


def extract_choice(text: str) -> str:
    # Final Answer: D
    m = re.search(r"final\s*answer\s*[:：]?\s*\(?\s*([A-D])\s*\)?", text, flags=re.I)
    if m: return m.group(1).upper()
    # Answer: D / Choice: D
    m = re.search(r"\b(answer|choice)\s*[:：]?\s*\(?\s*([A-D])\s*\)?", text, flags=re.I)
    if m: return m.group(2).upper()
    # Correct answer is D
    m = re.search(r"correct\s+answer\s+(is|:)?\s*\(?\s*([A-D])\s*\)?", text, flags=re.I)
    if m: return m.group(2).upper()

    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else ""


OPTION_LETTER_CTX = re.compile(
    r"(?ix)\b(?:final\s*answer|correct\s*answer|answer\s*(?:is|:))\b[^A-Z]*\(?\s*([A-D])\s*\)?"
)


def classify_errors(pred_choice: str, gold: str, valid_letters: tuple, model_text: str, is_correct: int):
    """
    Returns (missing, inconsistent, hallucinated), mutually exclusive.
    Logic:
      - If is_correct == 1 -> all 0s (no more error checks)
      - Priority: Missing > Invalid Label (considered inconsistent) > Hallucinated > Inconsistent (commonly incorrect answers)
    """
    if is_correct == 1:
        return 0, 0, 0

    # 1) Missing
    if not pred_choice:
        return 1, 0, 0

    # 2) Invalid label -> Inconsistent
    if pred_choice not in valid_letters:
        return 0, 1, 0

    # 3) Hallucinated（Invalid letters or conflicting multiple choices in the option context）
    letters_ctx = {m.group(1).upper() for m in OPTION_LETTER_CTX.finditer(model_text or "")}
    contradictions = {l for l in letters_ctx if l in valid_letters and l != pred_choice}
    if len(contradictions) >= 1:
        return 0, 0, 1

    # 4) Inconsistent
    if pred_choice != gold:
        return 0, 1, 0

    return 0, 0, 0


def build_prompt(question: str, options: List[str], k_shot: int = 0) -> Tuple[str, str]:
    letters = "ABCD"
    labeled = [f"{letters[i]}. {opt}" for i, opt in enumerate(options[:4])]
    system = "You are a board-certified physician. Answer USMLE-style multiple-choice questions correctly."

    examples = ""
    if k_shot == 1:
        examples += (
            "Example Question:\n"
            "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\n"
            "A. Ampicillin\nB. Ceftriaxonee\nC. Doxycycline\nD. Nitrofurantoin\n"
            "Final Answer: D\n\n"
        )
    if k_shot == 5:
        examples += (
            "Example Question:\n"
            "A 40-year-old zookeeper presents to the emergency department complaining of severe abdominal pain that radiates to her back, and nausea. The pain started 2 days ago and slowly increased until she could not tolerate it any longer. Past medical history is significant for hypertension and hypothyroidism. Additionally, she reports that she was recently stung by one of the zoo’s smaller scorpions, but did not seek medical treatment. She takes aspirin, levothyroxine, oral contraceptive pills, and a multivitamin daily. Family history is noncontributory. Today, her blood pressure is 108/58 mm Hg, heart rate is 99/min, respiratory rate is 21/min, and temperature is 37.0°C (98.6°F). On physical exam, she is a well-developed, obese female that looks unwell. Her heart has a regular rate and rhythm. Radial pulses are weak but symmetric. Her lungs are clear to auscultation bilaterally. Her lateral left ankle is swollen, erythematous, and painful to palpate. An abdominal CT is consistent with acute pancreatitis. Which of the following is the most likely etiology for this patient’s disease?\n"
            "A. Aspirin\nB. Oral contraceptive pills\nC. Scorpion sting\nD. Hypothyroidism\n"
            "Final Answer: C\n\n"
            "Example Question:\n"
            "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?\n"
            "A. Placing the infant in a supine position on a firm mattress while sleeping\nB. Keeping the infant covered and maintaining a high room temperature\nC. Application of a device to maintain the sleeping position\nD. Avoiding pacifier use during sleep\n"
            "Final Answer: A\n\n"
            "Example Question:\n"
            "A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?\n"
            "A. Abnormal migration of ventral pancreatic bud\nB. Complete failure of proximal duodenum to recanalize\nC. Abnormal hypertrophy of the pylorus\nD. Failure of lateral body folds to move ventrally and fuse in the midline\n"
            "Final Answer: A\n\n"
            "Example Question:\n"
            "A pulmonary autopsy specimen from a 58-year-old woman who died of acute hypoxic respiratory failure was examined. She had recently undergone surgery for a fractured femur 3 months ago. Initial hospital course was uncomplicated, and she was discharged to a rehab facility in good health. Shortly after discharge home from rehab, she developed sudden shortness of breath and had cardiac arrest. Resuscitation was unsuccessful. On histological examination of lung tissue, fibrous connective tissue around the lumen of the pulmonary artery is observed. Which of the following is the most likely pathogenesis for the present findings?\n"
            "A. Thromboembolism\nB. Pulmonary ischemia\nC. Pulmonary hypertension\nD. Pulmonary passive congestion\n"
            "Final Answer: A\n\n"
            "Example Question:\n"
            "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms?\n"
            "A. Hemophilia A\nB. Lupus anticoagulant\nC. Protein C deficiency\nD. Von Willebrand disease\n"
            "Final Answer: D\n\n"
        )

    user = (
        "Instructions: 1. Select exactly ONE option. 2. Output short rationale (<=30 words). 3. Final line must be: \"Final Answer: <LETTER>\". 4. <LETTER> must be one of: A/B/C/D"
        f"{examples}"
        f"{question.strip()}\n\nOptions:{os.linesep.join(labeled)}"
    )
    return system, user


# ---------- Data loading ----------
def _norm_txt(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .;:")
    return s.lower()


def _gold_text_to_letter(options, gold_text) -> str:
    opts_norm = [_norm_txt(o) for o in options[:4]]
    gold_norm = _norm_txt(gold_text)

    for i, on in enumerate(opts_norm):
        if on == gold_norm:
            return "ABCD"[i]

    for i, on in enumerate(opts_norm):
        if gold_norm in on or on in gold_norm:
            return "ABCD"[i]

    return ""


def load_medqa(max_samples=None, seed=42) -> pd.DataFrame:
    """
    MedQA (USMLE, English, 4-option) from BigBio.
    needs datasets<3.0。
    outputs：id, question, options(list[str]), gold(letter)
    """
    ds = load_dataset("bigbio/med_qa", name="med_qa_en_4options_bigbio_qa", split="test")

    records = []
    for ex in ds:
        qid = str(ex["id"])

        question = (ex["question"] or "").strip()
        ctx = (ex.get("context") or "").strip()
        if ctx:
            question = f"{ctx}\n\n{question}"

        options = list(ex["choices"])[:4]
        gold_text = str(ex["answer"]).strip()

        if not options or not gold_text:
            continue

        gold_letter = _gold_text_to_letter(options, gold_text)
        if gold_letter == "":
            continue

        records.append({
            "id": qid,
            "question": question,
            "options": options,
            "gold": gold_letter
        })

    df = pd.DataFrame(records)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


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
    VALID_LETTERS = ("A", "B", "C", "D")
    logs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model}"):
        sys_p, usr_p = build_prompt(row["question"], row["options"], k_shot=k_shot)
        resp = call_gpt(sys_p, usr_p, model=model)
        text = resp["text"]
        pred_choice = extract_choice(text)
        is_correct = 1 if (pred_choice == row["gold"]) else 0

        refusal = 0
        err_missing = 0
        err_inconsistent = 0
        err_hallucinated = 0

        if is_correct:
            pass
        else:
            if is_refusal(text):
                refusal = 1
            else:
                err_missing, err_inconsistent, err_hallucinated = classify_errors(
                    pred_choice=pred_choice,
                    gold=row["gold"],
                    valid_letters=VALID_LETTERS,
                    model_text=text,
                    is_correct=is_correct,
                )
        cost = estimate_cost_usd(model, resp["input_tokens"], resp["output_tokens"])
        logs.append(ItemLog(
            qid=row["id"], model=model, k_shot=k_shot, latency_ms=resp["latency_ms"],
            tokens_in=resp["input_tokens"], tokens_out=resp["output_tokens"], cost_usd=cost,
            gold=row["gold"], pred=pred_choice, is_correct=is_correct,
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


def main():
    df = load_medqa(max_samples=MAX_SAMPLES)
    # df.to_csv("/Users/hou00127/Google/Works/2025/Benchmarks/version_2/MedQA_USMLE/data.csv", index=False)
    # summaries = []
    # for model in MODELS:
    #     for k_shot in [0, 1, 5]:
    #         df_log, summary = eval_model_on_medqa(model, df, k_shot)
    #         df_log.to_csv(f"results_medqa_{model}_{k_shot}.csv", index=False)
    #         print(summary)
    #         summaries.append(summary)
    # with open("summary_medqa_k_shot.json", "w") as f: json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
