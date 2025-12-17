"""
_*_CODING:UTF-8_*_
@Author: Yu Hou (adapter by ChatGPT)
@File: PubMedQA.py
@Time: 2025-09-26
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate GPT-5 and GPT-4o on PubMedQA (yes/no/maybe).
This mirrors the structure and summary fields of your MedQA_USMLE.py, including:
  - call_gpt (latency + token usage)
  - is_refusal (pattern-based)
  - classify_errors (missing/inconsistent/hallucinated)
  - estimate_cost_usd (per-1K token pricing)
  - ItemLog dataclass (same fields)
  - eval_model_on_pubmedqa -> returns (df_log, summary) with the SAME keys used in eval_model_on_medqa:
        "model", "n", "accuracy", "avg_tokens_in", "avg_tokens_out",
        "total_cost_usd", "avg_latency_ms", "missing_rate",
        "inconsistent_rate", "hallucinated_rate", "refusal_rate"
Outputs:
  - results_pubmedqa_<model>.csv
  - summary_pubmedqa.json
"""

import os
import re
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

# If you are using the new OpenAI SDK
try:
    from openai import OpenAI

    OPENAI_API_KEY = ""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    client = None  # Will raise if call_gpt is used without proper setup

# ---------- Pricing per 1K tokens ----------
# Mirrors your MedQA script style; extend as needed.
PRICING_PER_1K = {
    "gpt-5": {"input": 0.005, "output": 0.015},  # seen in your MedQA file
    "gpt-4o": {"input": 0.0025, "output": 0.010},  # fill-in; adjust to your internal references
}

# ---------- Refusal detection ----------
REFUSAL_PATTERNS = [
    r"\bi('?m| am)\s(sorry|unable|not able)\b",
    r"\bcan(not|'t)\b\s(answer|comply|help)",
    r"\bas an ai\b",
]


def is_refusal(text: str) -> bool:
    return any(re.search(pat, text.lower()) for pat in REFUSAL_PATTERNS)


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


# ---------- Cost ----------
def estimate_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens / 1000.0) * p["input"] + (out_tokens / 1000.0) * p["output"]


# ---------- Model call ----------
def call_gpt(system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> Dict:
    if client is None:
        raise RuntimeError("OpenAI client is not initialized. Ensure your environment is configured.")
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
        return {
            "ok": False, "text": "", "latency_ms": (time.time() - start) * 1000.0,
            "input_tokens": 0, "output_tokens": 0, "error": str(e)
        }


# ---------- Prompt building for PubMedQA ----------
def build_prompt_pubmedqa(question: str, contexts: List[str], k_shot: int = 0) -> Tuple[str, str]:
    # Concatenate contexts; keep length moderate for cost
    ctx = "\n\n".join(c.strip() for c in contexts if c and c.strip())
    system = "You are a careful medical QA assistant. Answer PubMed-style yes/no/maybe questions accurately."
    examples = ""
    if k_shot == 1:
        examples += (
            "Example Question:Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
            "Context:Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.', 'The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.\n"
            "Final Answer: yes\n\n"
        )
    if k_shot == 5:
        examples += (
            "Example Question:Landolt C and snellen e acuity: differences in strabismus amblyopia?\n"
            "Context:Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize different optotypes differs even if their critical details appear under the same visual angle. Since optotypes are evaluated on individuals with good visual acuity and without eye disorders, differences in the lower visual acuity range cannot be excluded. In this study, visual acuity measured with the Snellen E was compared to the Landolt C acuity.', '100 patients (age 8 - 90 years, median 60.5 years) with various eye disorders, among them 39 with amblyopia due to strabismus, and 13 healthy volunteers were tested. Charts with the Snellen E and the Landolt C (Precision Vision) which mimic the ETDRS charts were used to assess visual acuity. Three out of 5 optotypes per line had to be correctly identified, while wrong answers were monitored. In the group of patients, the eyes with the lower visual acuity, and the right eyes of the healthy subjects, were evaluated.', 'Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small. The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia. The mean difference between LR and SE was 0.55 lines in the entire group and 0.55 lines for the eyes with strabismus amblyopia, with higher values of SE in both groups. The results of the other groups were similar with only small differences between LR and SE.\n"
            "Final Answer: no\n\n"
            "Example Question:Syncope during bathing in infants, a pediatric form of water-induced urticaria?\n"
            "Context:Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice. The prognosis is uncertain because of risk of sudden infant death syndrome.', 'Eight infants aged 2 to 15 months were admitted during a period of 6 years; they suffered from similar maladies in the bath: on immersion, they became pale, hypotonic, still and unreactive; recovery took a few seconds after withdrawal from the bath and stimulation. Two diagnoses were initially considered: seizure or gastroesophageal reflux but this was doubtful. The hypothesis of an equivalent of aquagenic urticaria was then considered; as for patients with this disease, each infant\'s family contained members suffering from dermographism, maladies or eruption after exposure to water or sun. All six infants had dermographism. We found an increase in blood histamine levels after a trial bath in the two infants tested. The evolution of these \"aquagenic maladies\" was favourable after a few weeks without baths. After a 2-7 year follow-up, three out of seven infants continue to suffer from troubles associated with sun or water.\n"    
            "Final Answer: yes\n\n"
            "Example Question:Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through?\n"
            "Context:The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment of Hirschsprung disease (HD), but overstretching of the anal sphincters remains a critical issue that may impact the continence. This study examined the long-term outcome of TERPT versus conventional transabdominal (ABD) pull-through for HD.', 'Records of 41 patients more than 3 years old who underwent a pull-through for HD (TERPT, n = 20; ABD, n = 21) were reviewed, and their families were thoroughly interviewed and scored via a 15-item post-pull-through long-term outcome questionnaire. Patients were operated on between the years 1995 and 2003. During this time, our group transitioned from the ABD to the TERPT technique. Total scoring ranged from 0 to 40: 0 to 10, excellent; 11 to 20 good; 21 to 30 fair; 31 to 40 poor. A 2-tailed Student t test, analysis of covariance, as well as logistic and linear regression were used to analyze the collected data with confidence interval higher than 95%.', 'Overall scores were similar. However, continence score was significantly better in the ABD group, and the stool pattern score was better in the TERPT group. A significant difference in age at interview between the 2 groups was noted; we therefore reanalyzed the data controlling for age, and this showed that age did not significantly affect the long-term scoring outcome between groups.\n"
            "Final Answer: no\n\n"
            "Example Question:Can tailored interventions increase mammography use among HMO women?\n"
            "Context:Telephone counseling and tailored print communications have emerged as promising methods for promoting mammography screening. However, there has been little research testing, within the same randomized field trial, of the efficacy of these two methods compared to a high-quality usual care system for enhancing screening. This study addressed the question: Compared to usual care, is tailored telephone counseling more effective than tailored print materials for promoting mammography screening?', 'Three-year randomized field trial.', 'One thousand ninety-nine women aged 50 and older recruited from a health maintenance organization in North Carolina.', 'Women were randomized to 1 of 3 groups: (1) usual care, (2) tailored print communications, and (3) tailored telephone counseling.', 'Adherence to mammography screening based on self-reports obtained during 1995, 1996, and 1997.', 'Compared to usual care alone, telephone counseling promoted a significantly higher proportion of women having mammograms on schedule (71% vs 61%) than did tailored print (67% vs 61%) but only after the first year of intervention (during 1996). Furthermore, compared to usual care, telephone counseling was more effective than tailored print materials at promoting being on schedule with screening during 1996 and 1997 among women who were off-schedule during the previous year."
            "Final Answer: yes\n\n"
            "Example Question:Double balloon enteroscopy: is it efficacious and safe in a community setting?\n"
            "Context:From March 2007 to January 2011, 88 DBE procedures were performed on 66 patients. Indications included evaluation anemia/gastrointestinal bleed, small bowel IBD and dilation of strictures. Video-capsule endoscopy (VCE) was used prior to DBE in 43 of the 66 patients prior to DBE evaluation.', \"The mean age was 62 years. Thirty-two patients were female, 15 were African-American; 44 antegrade and 44 retrograde DBEs were performed. The mean time per antegrade DBE was 107.4±30.0 minutes with a distance of 318.4±152.9 cm reached past the pylorus. The mean time per lower DBE was 100.7±27.3 minutes with 168.9±109.1 cm meters past the ileocecal valve reached. Endoscopic therapy in the form of electrocautery to ablate bleeding sources was performed in 20 patients (30.3%), biopsy in 17 patients (25.8%) and dilation of Crohn's-related small bowel strictures in 4 (6.1%). 43 VCEs with pathology noted were performed prior to DBE, with findings endoscopically confirmed in 32 cases (74.4%). In 3 cases the DBE showed findings not noted on VCE.\"\n"
            "Final Answer: yes\n\n"
        )

    user = (
        "Instructions: 1. Answer with exactly ONE word: \"yes\", \"no\", or \"maybe\". 2. Provide at most one short sentence of reasoning BEFORE the final line. 3. The FINAL line must be exactly: Final Answer: <yes|no|maybe>"
        f"{examples}"
        f"Question: {question.strip()}\nContext:{ctx}"
    )

    return system, user


# ---------- Extraction (yes/no/maybe) ----------
# 只在“最终答案语境”内提取 yes/no/maybe，避免把普通英文中的 "no" 误当标签
FINAL_ANSWER_REGEX = re.compile(
    r"(?i)\b(?:final\s*answer|correct\s*answer|answer\s*(?:is|:))\b[^a-zA-Z]*(yes|no|maybe)\b"
)


def extract_label_ynm(text: str) -> str:
    # Look specifically for Final Answer: yes/no/maybe
    m = re.search(r"final\s*answer\s*[:：]?\s*\(?\s*(yes|no|maybe)\s*\)?", text, flags=re.I)
    if m:
        return m.group(1).lower()

    # 如果没有明确 Final Answer，就返回空，不要 fallback
    return ""


# ---------- Error classification (mirrors logic of MedQA classify_errors) ----------
def classify_errors(pred_label: str, gold: str, valid_labels: tuple, model_text: str, is_correct: int):
    """
    Returns (missing, inconsistent, hallucinated), mutually exclusive.
    Logic (mirrors your MedQA):
      - If correct -> (0,0,0)
      - Priority: Missing > Invalid Label (inconsistent) > Hallucinated > Inconsistent (wrong but valid label)
    """
    if is_correct == 1:
        return 0, 0, 0

    # 1) Missing
    if not pred_label:
        return 1, 0, 0

    # 2) Invalid label -> Inconsistent
    if pred_label not in valid_labels:
        return 0, 1, 0

    # 3) Hallucinated: multiple conflicting labels in model text
    labels_final_ctx = {m.group(1).lower() for m in FINAL_ANSWER_REGEX.finditer(model_text or "")}
    contradictions = {l for l in labels_final_ctx if l in valid_labels and l != pred_label}
    if len(contradictions) >= 1:
        return 0, 0, 1  # hallucinated

    # 4) Inconsistent: single valid but wrong
    if pred_label != gold:
        return 0, 1, 0

    return 0, 0, 0


# ---------- Load PubMedQA ----------
def load_pubmedqa(path: str, max_samples: int = None, seed: int = 42) -> pd.DataFrame:
    """Read ori_pqal.json and produce a DataFrame with columns:
       id, question, contexts(list[str]), gold(str in {'yes','no','maybe'})
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for pmid, item in data.items():
        q = (item.get("QUESTION") or "").strip()
        ctxs = item.get("CONTEXTS") or []
        gold = (item.get("final_decision") or "").strip().lower()
        if not q or not gold:
            continue
        items.append({"id": str(pmid), "question": q, "contexts": ctxs, "gold": gold})

    if max_samples is not None:
        random.seed(seed)
        items = random.sample(items, k=min(max_samples, len(items)))

    df = pd.DataFrame(items, columns=["id", "question", "contexts", "gold"])
    return df


# ---------- Evaluation ----------
def eval_model_on_pubmedqa(model: str, df: pd.DataFrame, k_shot: int = 0):
    VALID = ("yes", "no", "maybe")
    logs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model} on PubMedQA"):
        sys_p, usr_p = build_prompt_pubmedqa(row["question"], row["contexts"], k_shot)
        resp = call_gpt(sys_p, usr_p, model=model)
        text = resp["text"]
        pred = extract_label_ynm(text)
        is_correct = 1 if (pred == row["gold"]) else 0

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
                    pred_label=pred,
                    gold=row["gold"],
                    valid_labels=VALID,
                    model_text=text,
                    is_correct=is_correct,
                )

        cost = estimate_cost_usd(model, resp.get("input_tokens", 0), resp.get("output_tokens", 0))
        logs.append(ItemLog(
            qid=row["id"],
            model=model,
            k_shot=k_shot,
            latency_ms=resp["latency_ms"],
            tokens_in=resp.get("input_tokens", 0),
            tokens_out=resp.get("output_tokens", 0),
            cost_usd=cost,
            gold=row["gold"],
            pred=pred,
            is_correct=is_correct,
            refusal=refusal,
            err_missing=err_missing,
            err_inconsistent=err_inconsistent,
            err_hallucinated=err_hallucinated,
            raw_output=text
        ))

    # to DataFrame
    df_log = pd.DataFrame([asdict(x) for x in logs])

    # SAME summary keys as eval_model_on_medqa
    summary = {
        "model": model,
        "k_shot": k_shot,
        "n": int(len(df_log)),
        "accuracy": round(float(df_log["is_correct"].mean()), 4) if len(df_log) else 0.0,
        "avg_tokens_in": round(float(df_log["tokens_in"].mean()), 2) if len(df_log) else 0.0,
        "avg_tokens_out": round(float(df_log["tokens_out"].mean()), 2) if len(df_log) else 0.0,
        "total_cost_usd": round(float(df_log["cost_usd"].sum()), 4) if len(df_log) else 0.0,
        "avg_latency_ms": round(float(df_log["latency_ms"].mean()), 2) if len(df_log) else 0.0,
        "missing_rate": round(float(df_log["err_missing"].mean()), 4) if len(df_log) else 0.0,
        "inconsistent_rate": round(float(df_log["err_inconsistent"].mean()), 4) if len(df_log) else 0.0,
        "hallucinated_rate": round(float(df_log["err_hallucinated"].mean()), 4) if len(df_log) else 0.0,
        "refusal_rate": round(float(df_log["refusal"].mean()), 4) if len(df_log) else 0.0,
    }
    return df_log, summary


# ---------- Default config & CLI ----------
MODELS = ["gpt-5", "gpt-4o"]
MAX_SAMPLES = None  # set None for full test
PUBMEDQA_JSON = "ori_pqal.json"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=PUBMEDQA_JSON, help="Path to ori_pqal.json")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="Subset size")
    args = parser.parse_args()

    df = load_pubmedqa(args.data, max_samples=args.max_samples)
    summaries = []
    for k_shot in [1, 5]:
        for model in args.models:
            df_log, summary = eval_model_on_pubmedqa(model, df, k_shot)
            out_csv = f"results_pubmedqa_{model}_{k_shot}.csv"
            df_log.to_csv(out_csv, index=False)
            print(summary)
            summaries.append(summary)
    with open("summary_pubmedqa_k_shot.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
