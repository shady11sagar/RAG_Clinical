import argparse, json
from rouge_score import rouge_scorer
from bert_score import score as bertscore

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--references", required=True)
    args = ap.parse_args()

    preds = [r["text"] for r in load_jsonl(args.predictions)]
    refs = [r["text"] for r in load_jsonl(args.references)]

    rouge_scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
    rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
    rouge2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores)
    rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)

    P, R, F1 = bertscore(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli")

    print(json.dumps({
        "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL,
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }, indent=2))


if __name__ == "__main__":
    main()
