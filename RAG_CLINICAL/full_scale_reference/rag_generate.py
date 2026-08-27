import argparse, json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEN_PROMPT = (
    "### De-identified Clinical Input:\n{input}\n\n"
    "### Retrieved Clinical Guidance (cite by id in your Plan):\n{context}\n\n"
    "### Structured SOAP Note:\n"
)


class RagGenerator:
    def __init__(self, base_model, lora_dir, index_path, meta_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto",
                                                      torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, lora_dir)
        self.model.eval()

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.meta = json.load(f)

    def retrieve(self, query, k=3):
        qv = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(qv, k)
        return [{**self.meta[i], "score": float(s)} for s, i in zip(scores[0], idx[0])]

    def generate(self, deid_input, k=3, max_new_tokens=400):
        retrieved = self.retrieve(deid_input, k=k)
        context = "\n".join(f"[{r['id']}] {r['text']}" for r in retrieved)
        prompt = GEN_PROMPT.format(input=deid_input, context=context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False, temperature=0.0)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        return {"note": text, "sources": retrieved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--input_text", required=True)
    args = ap.parse_args()

    gen = RagGenerator(args.base_model, args.lora_dir, args.index, args.meta)
    result = gen.generate(args.input_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
