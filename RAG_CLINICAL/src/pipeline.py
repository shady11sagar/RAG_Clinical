import os, json, re
from deid import deidentify
from section_classifier import classify_sentences
from rag_retrieval import GuidelineRetriever

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

GENERIC_BOILERPLATE_PLAN = "Continue monitoring symptoms and follow up as needed."
GENERIC_BOILERPLATE_ASSESSMENT = "Findings reviewed; clinical picture appears stable at this time."


def extract_subjective_objective(transcript):
    """Phase I (de-id) + classifier-based extraction from a raw transcript."""
    lines = [turn["text"] for turn in transcript]
    deid_lines = [deidentify(l)[0] for l in lines]
    preds = classify_sentences(deid_lines)
    subj = [l for l, p in zip(deid_lines, preds) if p == "Subjective"]
    obj = [l for l, p in zip(deid_lines, preds) if p == "Objective"]
    return subj, obj


def generate_note(encounter, retriever: GuidelineRetriever, use_rag: bool, k=2):
    subj, obj = extract_subjective_objective(encounter["transcript"])
    query = " ".join(subj + obj) if (subj or obj) else " ".join(
        [t["text"] for t in encounter["transcript"]])

    if use_rag:
        retrieved = retriever.retrieve(query, k=k)
        top = retrieved[0] if retrieved else None
        if top:
            assessment = f"Clinical impression is consistent with {top['condition'].replace('_', ' ')}, per retrieved guidance [{top['id']}]."
            plan = f"{top['text']} [source: {top['id']}]"
        else:
            assessment, plan = GENERIC_BOILERPLATE_ASSESSMENT, GENERIC_BOILERPLATE_PLAN
        sources = retrieved
    else:
        assessment, plan = GENERIC_BOILERPLATE_ASSESSMENT, GENERIC_BOILERPLATE_PLAN
        sources = []

    return {
        "id": encounter["id"],
        "condition": encounter["condition"],
        "subjective": subj,
        "objective": obj,
        "assessment": assessment,
        "plan": plan,
        "sources": sources,
    }


def run_pipeline_on_split(split="test", use_rag=True, k=2):
    with open(os.path.join(DATA, "encounters.json")) as f:
        encounters = json.load(f)[split]
    retriever = GuidelineRetriever()
    notes = [generate_note(e, retriever, use_rag=use_rag, k=k) for e in encounters]
    return encounters, notes


if __name__ == "__main__":
    encounters, notes = run_pipeline_on_split("test", use_rag=True)
    print(json.dumps(notes[0], indent=2))
