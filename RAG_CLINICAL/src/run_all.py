import subprocess, sys, os

HERE = os.path.dirname(__file__)
STEPS = [
    "generate_data.py",
    "deid.py",
    "section_classifier.py",
    "rag_retrieval.py",
    "evaluate.py",
    "adversarial_test.py",
    "make_figures.py",
    "make_architecture_diagram.py",
    "make_pipeline_diagnostics_diagram.py",
    "make_rouge_groundedness_tradeoff_diagram.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'='*60}\nRunning {step}\n{'='*60}")
        r = subprocess.run([sys.executable, step], cwd=HERE)
        if r.returncode != 0:
            print(f"FAILED at {step}")
            sys.exit(1)
    print("\nAll experiments completed. See ../results/ and ../figures/.")
