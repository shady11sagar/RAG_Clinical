import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
MODELS = os.path.join(HERE, "..", "results")
os.makedirs(MODELS, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA, "sentences_labeled.csv"))
    return df


def train_and_evaluate(seed=42):
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=seed, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=5.0)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, output_dict=True)
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, preds, labels=labels)

    joblib.dump({"vectorizer": vectorizer, "clf": clf, "labels": labels},
                os.path.join(MODELS, "section_classifier.joblib"))

    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(MODELS, "section_classifier_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_model():
    return joblib.load(os.path.join(MODELS, "section_classifier.joblib"))


def classify_sentences(sentences):
    bundle = load_model()
    X = bundle["vectorizer"].transform(sentences)
    preds = bundle["clf"].predict(X)
    return list(preds)


if __name__ == "__main__":
    res = train_and_evaluate()
    print(f"Accuracy: {res['accuracy']:.3f}  Macro-F1: {res['macro_f1']:.3f}")
    print(f"Train/Test size: {res['n_train']}/{res['n_test']}")
