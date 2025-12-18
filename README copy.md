
## Installation

```bash
git clone https://github.com/yourusername/Benchmark_GPT5.git
cd Benchmark_GPT5
pip install -r requirements.txt
```

## Project Structure

```
Benchmark_GPT5/
├── src/
│   ├── dataset_manager/
│   ├── model_manager/
│   ├── prompt_manager/
│   └── batch_api_manager/
├── scripts/
│   ├── run_inference.py
│   └── run_batch_api.py
└── requirements.txt
```

## Usage

### Local Inference

```bash
python scripts/run_inference.py \
  --dataset <dataset_name> \
  --model <model_name> \
  --gen \
  --eval \
  --quantization 4bit \
  --max_new_tokens 512
```

### Batch Processing

```bash
DATASETS=(ddi bc5cdr chemprot ncbi hoc litcovid medqa pubmedqa cochranepls plos ms2 pubmed)

for dataset in "${DATASETS[@]}"; do
  python scripts/run_batch_api.py \
    --action submit \
    --dataset "$dataset" \
    --model gpt-4o \
    --description "$dataset evaluation" &
  
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do
    sleep 1
  done
done

wait
```

### Few-Shot Learning

```bash
for num_shots in 1 5; do
  python scripts/run_batch_api.py \
    --action submit \
    --dataset bc5cdr \
    --model gpt-4o \
    --few_shot $num_shots \
    --description "BC5CDR ${num_shots}-shot"
done
```

### Batch API

```bash
export OPENAI_API_KEY="your-api-key"

python scripts/run_batch_api.py \
  --action submit \
  --dataset <dataset_name> \
  --model gpt-4o \
  --description "Your description"

python scripts/run_batch_api.py --action check --batch_id <batch_id>

python scripts/run_batch_api.py --action retrieve --batch_id <batch_id>
```

## Tasks

- **Named Entity Recognition (NER)**: BC5CDR, NCBI Disease
- **Relation Extraction (RE)**: ChemProt, DDI
- **Question Answering (QA)**: MedQA, PubMedQA
- **Multi-Label Classification (MLC)**: HoC, LitCovid
- **Simplification**: CochranePLS, PLOS
- **Summarization**: MS2, PubMed