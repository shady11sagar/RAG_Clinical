import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline import run_pipeline_on_split
from rag_retrieval import GuidelineRetriever

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def note_to_text(note_or_gold, is_gold=False):
    if is_gold:
        return " ".join(note_or_gold["subjective"] + note_or_gold["objective"]
                         + [note_or_gold["assessment"], note_or_gold["plan"]])
    return " ".join(note_or_gold["subjective"] + note_or_gold["objective"]
                     + [note_or_gold["assessment"], note_or_gold["plan"]])


def rouge_for_pair(gold_text, gen_text):
    scores = scorer.score(gold_text, gen_text)
    return {k: v.fmeasure for k, v in scores.items()}


def groundedness(plan_text, retriever):
    """Cosine similarity between the generated Plan text and its single
    closest guideline passage — a proxy for how traceable/source-grounded
    a generated clinical claim is."""
    vec = TfidfVectorizer().fit(retriever.texts + [plan_text])
    m = vec.transform(retriever.texts)
    q = vec.transform([plan_text])
    sims = cosine_similarity(q, m)[0]
    return float(np.max(sims)) if len(sims) else 0.0


def run_condition(use_rag: bool):
    encounters, notes = run_pipeline_on_split("test", use_rag=use_rag, k=2)
    retriever = GuidelineRetriever()

    rouge_rows, ground_rows, condition_hits = [], [], []
    for enc, note in zip(encounters, notes):
        gold_text = note_to_text(enc["gold_soap"], is_gold=True)
        gen_text = note_to_text(note)
        rouge_rows.append(rouge_for_pair(gold_text, gen_text))
        ground_rows.append(groundedness(note["plan"], retriever))
        if use_rag:
            predicted_condition = note["sources"][0]["condition"] if note["sources"] else None
            condition_hits.append(int(predicted_condition == enc["condition"]))

    agg = {
        "rouge1": float(np.mean([r["rouge1"] for r in rouge_rows])),
        "rouge2": float(np.mean([r["rouge2"] for r in rouge_rows])),
        "rougeL": float(np.mean([r["rougeL"] for r in rouge_rows])),
        "groundedness_mean": float(np.mean(ground_rows)),
        "n": len(encounters),
    }
    if use_rag:
        agg["end_to_end_condition_accuracy"] = float(np.mean(condition_hits))
    return agg, rouge_rows, ground_rows


def main():
    rag_agg, rag_rouge, rag_ground = run_condition(use_rag=True)
    norag_agg, norag_rouge, norag_ground = run_condition(use_rag=False)

    results = {"rag": rag_agg, "no_rag": norag_agg}
    with open(os.path.join(RES, "generation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # --- Figure 1: ROUGE comparison bar chart ---
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    rag_vals = [rag_agg["rouge1"], rag_agg["rouge2"], rag_agg["rougeL"]]
    norag_vals = [norag_agg["rouge1"], norag_agg["rouge2"], norag_agg["rougeL"]]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(x - w/2, norag_vals, w, label="No-RAG baseline", color="#a8a8a8")
    ax.bar(x + w/2, rag_vals, w, label="RAG-augmented (proposed)", color="#1f3864")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("F-measure")
    ax.set_title("Generated Note Quality vs. Gold SOAP Note")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rouge_comparison.png"), dpi=200)
    plt.close(fig)

    # --- Figure 2: Groundedness comparison ---
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.bar(["No-RAG baseline", "RAG-augmented"],
           [norag_agg["groundedness_mean"], rag_agg["groundedness_mean"]],
           color=["#a8a8a8", "#1f3864"])
    ax.set_ylabel("Mean cosine similarity to nearest guideline")
    ax.set_title("Groundedness of Generated Plan Statements")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_groundedness.png"), dpi=200)
    plt.close(fig)

    return results


if __name__ == "__main__":
    main()
