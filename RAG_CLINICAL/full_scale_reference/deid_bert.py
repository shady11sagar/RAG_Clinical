
import argparse, json
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# For clinical-grade NER, swap the default spaCy/transformers pipeline for a
# clinically fine-tuned model, e.g. "obi/deid_roberta_i2b2" via
# transformers' token-classification pipeline registered with Presidio's
# NlpEngine. See Presidio docs for custom NlpEngine registration.

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

ENTITIES = ["PERSON", "PHONE_NUMBER", "US_SSN", "DATE_TIME", "LOCATION",
            "MEDICAL_LICENSE", "US_DRIVER_LICENSE"]


def deidentify_text(text: str) -> str:
    results = analyzer.analyze(text=text, entities=ENTITIES, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            rec["text"] = deidentify_text(rec["text"])
            fout.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
