import re, json, os

PHONE_RE = re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MRN_RE = re.compile(r"\bMRN\s?\d{6}\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
ADDRESS_RE = re.compile(
    r"\b\d{2,5}\s+\w+\s+(?:St|Ave|Rd|Ln)\.?,\s*\w+,\s*[A-Z]{2}\b"
)

NAME_RE = re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b")

# Names that are NOT PHI in this synthetic domain (guideline / drug-class
# terms that happen to be capitalized) — exclude to reduce false positives.
NAME_EXCLUDE = {"Blood Pressure", "Type Diabetes", "Straight Leg", "Peak Flow"}

TAG = "[REDACTED]"

def deidentify(text: str):
    spans_found = []
    def _sub(pattern, s, label):
        nonlocal spans_found
        for m in pattern.finditer(s):
            spans_found.append((m.group(), label))
        return pattern.sub(TAG, s)

    out = text
    out = _sub(ADDRESS_RE, out, "ADDRESS")
    out = _sub(SSN_RE, out, "SSN")
    out = _sub(MRN_RE, out, "MRN")
    out = _sub(PHONE_RE, out, "PHONE")
    out = _sub(DATE_RE, out, "DATE")

    # name pass, after structured identifiers are already masked
    def name_sub(m):
        cand = m.group()
        if cand in NAME_EXCLUDE:
            return cand
        spans_found.append((cand, "NAME"))
        return TAG
    out = NAME_RE.sub(name_sub, out)
    return out, spans_found


def evaluate(phi_test_path, verbose=False):
    with open(phi_test_path) as f:
        examples = json.load(f)

    tp = fp = fn = 0
    for ex in examples:
        redacted, found = deidentify(ex["text"])
        found_strings = set(s for s, _ in found)
        gold = set(ex["phi_spans"])
        tp += len(found_strings & gold)
        fp += len(found_strings - gold)
        fn += len(gold - found_strings)
        if verbose:
            print(ex["text"], "->", redacted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "phi_test.json")
    metrics = evaluate(path)
    print(json.dumps(metrics, indent=2))
    res_dir = os.path.join(here, "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "deid_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
