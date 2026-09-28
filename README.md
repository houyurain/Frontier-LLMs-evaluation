# Frontier LLMs Evaluation in Biomedical Tasks

This repository contains research code accompanying the manuscript:

> **Auditing frontier general-purpose large language models in biomedical tasks: reasoning gains, extraction limits, and benchmark reliability**

The study evaluates GPT-5 and GPT-4o across 19 biomedical datasets covering biomedical question answering, named entity recognition, relation extraction, multi-label classification, summarization, and text simplification.

## Repository scope

The nine top-level Python files in this repository correspond to the nine biomedical question-answering datasets evaluated in the study.

The other ten core BioNLP datasets were organized through shared task-family pipelines rather than separate dataset-specific scripts:

- six extraction and classification datasets shared an extraction/classification pipeline;
- four summarization and simplification datasets shared a generation pipeline.

The task definitions, prompts, dataset processing, and historical evaluation framework for these ten datasets were derived from the publicly available [Biomedical-NLP-Benchmarks repository](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks) of Chen et al.

The present repository documents the relationship between the 19 datasets and their corresponding code paths. It does not claim to provide a complete executable reconstruction of every historical GPT-5/GPT-4o Batch API request.

## Python file map

| File | Dataset | Purpose | Current entry-point behavior |
|---|---|---|---|
| `DiagnosisArena.py` | DiagnosisArena | Loads diagnostic multiple-choice cases, constructs zero- or few-shot prompts, calls the configured models, extracts final-answer labels, and reports accuracy and item-level execution records. | Command-line evaluation interface. |
| `MedCaseReasoning.py` | MedCaseReasoning | Evaluates diagnostic reasoning from clinical case reports and compares generated diagnoses with reference diagnoses using normalized exact, token-overlap, and alias-aware matching. | Command-line evaluation interface. |
| `MedMCQA.py` | MedMCQA | Contains data loading, prompting, model-calling, scoring, error classification, and subject-level analysis utilities for medical multiple-choice questions. | The current command-line entry point exposes result analysis; evaluation functions are defined in the file. |
| `MedQA_USMLE.py` | MedQA (USMLE) | Loads the English four-option MedQA test set, evaluates the configured models under zero-, one-, and five-shot settings, and reports answer accuracy and item-level execution records. | Runs the configured evaluation when executed. |
| `MedXpertQA.py` | MedXpertQA | Evaluates expert-level multiple-choice questions. The manuscript analysis uses the text subset; the script also contains support for the multimodal subset. | Command-line evaluation interface. |
| `PMC_VQA.py` | PMC-VQA | Loads image-based multiple-choice questions, prepares the required image subset, constructs multimodal prompts, and provides evaluation and scoring functions. | The current command-line entry point prepares data or exports metadata; the evaluation loop is disabled by default. |
| `PubMedQA.py` | PubMedQA | Evaluates evidence-grounded yes/no/maybe questions, extracts categorical answers, and reports accuracy and item-level execution records. | The current entry point runs the configured one- and five-shot analyses. |
| `SLAKE.py` | SLAKE | Loads English closed-form medical visual questions, constructs multimodal prompts, and provides scoring and result-analysis functions. | The current entry point runs result analysis; the evaluation loop is disabled by default. |
| `VQA_RAD.py` | VQA-RAD | Provides multimodal prompting, answer extraction, scoring, and result-analysis utilities for open- and closed-form radiology questions. | The current command-line entry point exposes result analysis; evaluation functions are defined in the file. |

## Dataset-to-code map

### Biomedical question answering

| Task | Dataset | Code |
|---|---|---|
| Medical multiple-choice QA | MedQA (USMLE) | `MedQA_USMLE.py` |
| Biomedical evidence QA | PubMedQA | `PubMedQA.py` |
| Medical multiple-choice QA | MedMCQA | `MedMCQA.py` |
| Expert-level medical QA | MedXpertQA | `MedXpertQA.py` |
| Diagnostic reasoning | DiagnosisArena | `DiagnosisArena.py` |
| Diagnostic reasoning | MedCaseReasoning | `MedCaseReasoning.py` |
| Medical visual QA | PMC-VQA | `PMC_VQA.py` |
| Medical visual QA | SLAKE | `SLAKE.py` |
| Radiology visual QA | VQA-RAD | `VQA_RAD.py` |

### Core BioNLP tasks

| Task family | Datasets | Shared framework |
|---|---|---|
| Named entity recognition | BC5CDR-Chemical; NCBI-Disease | [`GPT/extractive_tasks`](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT/extractive_tasks) |
| Relation extraction | ChemProt; DDI2013 | [`GPT/extractive_tasks`](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT/extractive_tasks) |
| Multi-label classification | HoC; LitCovid | [`GPT/extractive_tasks`](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT/extractive_tasks) |
| Biomedical summarization | PubMed; MS² | [`GPT/generative_tasks`](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT/generative_tasks) |
| Biomedical text simplification | Cochrane PLS; PLOS | [`GPT/generative_tasks`](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT/generative_tasks) |

## Why are there nine Python files for 19 datasets?

The repository is organized by evaluation workflow rather than by requiring one independent script for every dataset.

The nine top-level files are dataset-specific utilities for the nine QA datasets. The other ten datasets were evaluated through two shared task-family workflows:

1. **Extraction/classification workflow**

   - BC5CDR-Chemical
   - NCBI-Disease
   - ChemProt
   - DDI2013
   - HoC
   - LitCovid

2. **Generation workflow**

   - PubMed summarization
   - MS² summarization
   - Cochrane plain-language simplification
   - PLOS text simplification

Therefore, the number of datasets does not equal the number of Python files.

## Installation

Create an isolated Python environment before installing the dependencies.

```bash
python -m venv .venv
```

Activate the environment.

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies used by the public scripts:

```bash
pip install openai pandas tqdm "datasets<3.0"
```

The `datasets<3.0` constraint is required by the current MedQA/BigBio loader. Dataset-provider changes may require future loader updates.

## API configuration

Set the OpenAI API key through an environment variable. Do not place an actual API key in a Python file or commit it to Git.

On Linux or macOS:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

The scripts do not automatically load a `.env` file.

Some archived research scripts contain placeholder API-key assignments such as `"xx"` or `"xxxxx"`. These placeholders are not valid credentials. Before running those scripts, replace only the placeholder initialization with environment-variable loading:

```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)
```

Never replace a placeholder with a real credential inside tracked source code.

## Data access

This repository does not redistribute benchmark datasets or medical images.

Users are responsible for obtaining each dataset from its original source and complying with its licence, access requirements, and terms of use.

Several text datasets are loaded through Hugging Face:

- DiagnosisArena: `shzyk/DiagnosisArena`
- MedCaseReasoning: `zou-lab/MedCaseReasoning`
- MedQA: `bigbio/med_qa`
- MedMCQA: `lighteval/med_mcqa`
- MedXpertQA: `TsinghuaC3I/MedXpertQA`

Other scripts expect local files:

- `PubMedQA.py` expects an `ori_pqal.json` file or a path supplied through `--data`.
- `PMC_VQA.py` expects `test_clean.csv` and the associated image files under the directory specified by `PMC_VQA_DIR`.
- `SLAKE.py` expects the official SLAKE JSON files and images under the directory specified by `SLAKE_DIR`.
- `VQA_RAD.py` contains utilities for the VQA-RAD data and uses the paths defined in the script or supplied to its analysis interface.

Example environment-variable configuration:

```bash
export PMC_VQA_DIR="/path/to/PMC_VQA"
export SLAKE_DIR="/path/to/SLAKE"
```

Windows PowerShell:

```powershell
$env:PMC_VQA_DIR="C:\path\to\PMC_VQA"
$env:SLAKE_DIR="C:\path\to\SLAKE"
```

## Running the scripts

These files are research scripts rather than a unified command-line package. Their entry points differ, and some files expose analysis or data-preparation functions rather than automatically initiating paid API evaluations.

Review the model list, dataset split, sample limit, shot setting, and output directory before running any script. API calls may incur provider charges.

### DiagnosisArena

Display the available options:

```bash
python DiagnosisArena.py --help
```

Example zero-shot evaluation:

```bash
python DiagnosisArena.py --models gpt-5,gpt-4o --kshot 0
```

### MedCaseReasoning

Display the available options:

```bash
python MedCaseReasoning.py --help
```

Example zero-shot evaluation:

```bash
python MedCaseReasoning.py --model both --kshot 0
```

### MedQA (USMLE)

The current entry point evaluates the models and shot settings configured inside the script:

```bash
python MedQA_USMLE.py
```

### MedXpertQA text subset

Display the available options:

```bash
python MedXpertQA.py --help
```

Example zero-shot evaluation:

```bash
python MedXpertQA.py \
  --subset Text \
  --split test \
  --models gpt-5,gpt-4o \
  --kshot 0
```

### PubMedQA

```bash
python PubMedQA.py \
  --data /path/to/ori_pqal.json \
  --models gpt-5 gpt-4o
```

The current entry point runs the one- and five-shot conditions defined in the script.

### PMC-VQA data preparation

```bash
python PMC_VQA.py \
  --prepare-subset \
  --source-images /path/to/PMC_VQA/images
```

The evaluation functions are defined in the file, but the paid API evaluation loop is disabled by default.

### Analysis-oriented entry points

The current MedMCQA and VQA-RAD command-line interfaces expose result analysis:

```bash
python MedMCQA.py --help
python VQA_RAD.py --help
```

Example:

```bash
python MedMCQA.py \
  --analyze \
  --data-folder /path/to/MedMCQA/results \
  --analysis-model gpt-4o
```

```bash
python VQA_RAD.py \
  --analyze \
  --data-folder /path/to/VQA_RAD/results \
  --analysis-model gpt-5
```

The current SLAKE entry point also performs result analysis; its API evaluation loop is retained in the source but disabled by default.

## Outputs

Depending on the script, outputs may include:

- item-level CSV files;
- aggregate JSON summaries;
- predicted and reference answers;
- correctness indicators;
- refusal and output-format indicators;
- provider-reported token fields;
- client-observed elapsed time;
- script-estimated cost fields.

These fields must be interpreted carefully:

- elapsed time is client-observed end-to-end duration, not hardware-normalized or intrinsic model latency;
- token fields reflect the usage categories returned by the relevant API endpoint and may not be directly comparable across endpoints;
- estimated cost is calculated from price constants stored in the script and is not an authoritative billing record;
- provider prices and model behavior may change over time.

## Random seed and reproducibility

The scripts generally use a local random seed of `42` for dataset subsampling and/or few-shot example selection.

This seed controls local sampling only. It is not an API-level generation seed and does not guarantee identical model outputs across repeated calls.

Additional sources of variation include:

- changes to provider-side model aliases or snapshots;
- API and SDK updates;
- unspecified provider-default generation parameters;
- transient API failures or retries;
- dataset revisions;
- differences in local preprocessing or image availability.

For a new evaluation, record at minimum:

- execution date;
- configured and resolved model identifiers;
- dataset version and split;
- evaluated item identifiers;
- prompt or request hash;
- shot condition;
- API route;
- explicitly supplied generation parameters;
- package versions;
- technical failures and retry handling.

## Relationship to the historical benchmark

The core BioNLP component builds on:

> Chen et al. *Benchmarking large language models for biomedical natural language processing applications and recommendations*. Nature Communications 16, 3280 (2025).

Public resources from the prior benchmark include:

- [datasets and prompt templates](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/benchmarks);
- [GPT inference framework](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT);
- [evaluation scripts](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/blob/main/run_eval.py);
- [NER evaluation script](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/blob/main/run_eval_ner.py).

The historical GPT-4 and GPT-3.5 baseline results reported in the accompanying manuscript were inherited from that prior benchmark rather than rerun in the present repository.

## Reproducibility boundary

This repository provides the nine QA workflow scripts and documents the relationship between all 19 manuscript datasets and their corresponding evaluation framework.

It should not be interpreted as:

- a containerized or one-command reproduction package;
- a release of restricted dataset files or medical images;
- a complete executable reconstruction of every historical GPT-5/GPT-4o Batch API request;
- evidence that provider-side model aliases will resolve to the same snapshots in future runs;
- a guarantee that rerunning the scripts will produce numerically identical outputs.

The manuscript results should be interpreted together with the reported dataset splits, prompts, metrics, model identifiers, and stated reproducibility limitations.

## Security

Do not commit:

- API keys or other credentials;
- `.env` files;
- cloud billing or account identifiers;
- restricted datasets or medical images;
- raw responses containing private or restricted information;
- journal-review correspondence.

If a credential has ever been committed, removing it from the latest file is not sufficient. Revoke or rotate it and review the repository history before publication.

## Citation

If you use this repository, please cite the accompanying manuscript and the original benchmark framework. Full citation information for the accompanying manuscript will be added after publication.

## Questions

For questions about the public code, please open an issue in this repository.
