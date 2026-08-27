
import argparse, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # swap for a clinical
                                                          # embedding model
                                                          # (e.g. MedCPT) in
                                                          # production


def load_docs(path):
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--index_out", default="guideline.index")
    ap.add_argument("--meta_out", default="guideline_meta.json")
    args = ap.parse_args()

    docs = load_docs(args.docs)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                               show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via inner product
    index.add(embeddings)
    faiss.write_index(index, args.index_out)

    with open(args.meta_out, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"Indexed {len(docs)} guideline passages -> {args.index_out}")


if __name__ == "__main__":
    main()
