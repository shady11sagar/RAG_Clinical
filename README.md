# Secure RAG-Enhanced Clinical Documentation Framework


It is split into two parts, because the results reported in the paper were
produced in a sandboxed environment with **no GPU and no internet access to
model hubs** (Hugging Face, etc.), and with **no MIMIC-IV credentialed
access**. Rather than fabricate large-scale LLM results that were never run,
the paper reports real numbers from a small-scale, fully-synthetic,
CPU-only reference implementation, and provides GPU-ready code for
reproducing the full-scale design once real data/compute are available.

## `src/` — Small-scale reference implementation (what the paper's results come from)

Runs end-to-end on any laptop, no GPU, no internet access beyond `pip`.
Uses a **synthetic** dataset (`generate_data.py` creates it. No real
patient data anywhere in this repo) and classical ML/NLP
(TF-IDF + logistic regression, TF-IDF retrieval, regex de-identification)
as CPU-runnable stand-ins for the GPU components specified in the full
architecture.

> Before running any of this, create and activate a virtual environment, then make sure your editor (VS Code, Antigravity, etc.) has that same virtual environment selected as its interpreter, otherwise pip install and python run_all.py may end up pointing at two different Pythons.

```bash
cd src
pip install -r ../requirements.txt
python run_all.py          # regenerates data, trains, evaluates, plots — reproduces every number and figure in the paper
```

Outputs land in `../results/*.json` and `../figures/*.png`.

| File | What it does |
|---|---|
| `generate_data.py` | Builds the synthetic encounters, guideline KB, PHI test set |
| `deid.py` | Regex-based PHI de-identification + precision/recall/F1 |
| `section_classifier.py` | TF-IDF + Logistic Regression SOAP-section classifier |
| `rag_retrieval.py` | TF-IDF retrieval over guidelines + Precision@k |
| `pipeline.py` | Full note-generation pipeline (RAG vs. no-RAG) |
| `evaluate.py` | ROUGE, groundedness, RAG-vs-baseline comparison + charts |
| `adversarial_test.py` | Prompt-injection / PHI-leakage robustness probes |
| `make_figures.py`, `make_architecture_diagram.py` | Paper figures |

## `full_scale_reference/` — GPU-ready code for real deployment

Real, complete implementations of the components the small-scale version
stands in for: QLoRA fine-tuning of Llama 3 / Mistral (`train_lora.py`),
dense FAISS retrieval with sentence embeddings (`build_faiss_index.py`),
retrieval-augmented generation (`rag_generate.py`), hybrid Presidio + BERT-NER
de-identification (`deid_bert.py`), and ROUGE + real BERTScore evaluation
(`evaluate_full.py`). 

```bash
cd full_scale_reference
pip install -r requirements.txt
python train_lora.py --train_file data/train_deid.jsonl --base_model mistralai/Mistral-7B-Instruct-v0.3
python build_faiss_index.py --docs guidelines.jsonl
python rag_generate.py --lora_dir checkpoints/clinical-lora --index guideline.index --meta guideline_meta.json --input_text "..."
python evaluate_full.py --predictions preds.jsonl --references refs.jsonl
```


## No real patient data

Every dataset in `src/data/` (created by `generate_data.py`) is synthetic —
procedurally generated dialogue and notes with fictional names and
randomized vitals. Nothing here is, or is derived from, a real patient
record.
