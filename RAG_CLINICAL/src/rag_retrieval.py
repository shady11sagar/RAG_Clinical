import os, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")


class GuidelineRetriever:
    def __init__(self, guidelines_path=None):
        guidelines_path = guidelines_path or os.path.join(DATA, "guidelines.json")
        with open(guidelines_path) as f:
            self.guidelines = json.load(f)
        self.texts = [g["text"] for g in self.guidelines]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=2):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        return [
            {"id": self.guidelines[i]["id"], "condition": self.guidelines[i]["condition"],
             "text": self.guidelines[i]["text"], "score": float(sims[i])}
            for i in top_idx
        ]


def evaluate_retrieval(k=2):
    """Precision@k: for each encounter, does retrieving on the patient's
    subjective+objective text surface at least one guideline actually tagged
    to that encounter's true condition, within the top-k results?"""
    with open(os.path.join(DATA, "encounters.json")) as f:
        enc = json.load(f)
    all_enc = enc["train"] + enc["test"]
    retriever = GuidelineRetriever()

    hits, total = 0, 0
    per_condition = {}
    for e in all_enc:
        query = " ".join(e["gold_soap"]["subjective"] + e["gold_soap"]["objective"])
        results = retriever.retrieve(query, k=k)
        correct = any(r["condition"] == e["condition"] for r in results)
        hits += int(correct)
        total += 1
        pc = per_condition.setdefault(e["condition"], [0, 0])
        pc[0] += int(correct)
        pc[1] += 1

    precision_at_k = hits / total
    breakdown = {c: v[0] / v[1] for c, v in per_condition.items()}
    return {"precision_at_k": precision_at_k, "k": k, "n": total, "per_condition": breakdown}


if __name__ == "__main__":
    res = evaluate_retrieval(k=2)
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "..", "results", "retrieval_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
