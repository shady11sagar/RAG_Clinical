import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)

# ---- Confusion matrix for section classifier ----
with open(os.path.join(RES, "section_classifier_metrics.json")) as f:
    cm_data = json.load(f)
cm = np.array(cm_data["confusion_matrix"])
labels = cm_data["labels"]

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"SOAP-Section Classifier Confusion Matrix\n(Accuracy={cm_data['accuracy']:.3f}, Macro-F1={cm_data['macro_f1']:.3f})")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_confusion_matrix.png"), dpi=200)
plt.close(fig)

# ---- Retrieval precision@k by condition ----
with open(os.path.join(RES, "retrieval_metrics.json")) as f:
    ret = json.load(f)
conds = list(ret["per_condition"].keys())
vals = [ret["per_condition"][c] for c in conds]
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(conds, vals, color="#2e5aa8")
ax.set_ylim(0, 1.05)
ax.set_ylabel(f"Precision@{ret['k']}")
ax.set_title(f"Oracle Retrieval Precision@{ret['k']} by Condition (overall={ret['precision_at_k']:.2f})")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_retrieval_precision.png"), dpi=200)
plt.close(fig)

# ---- Equation images (mathtext rendered) ----
def render_eq(name, tex, fontsize=20, figsize=(6.5, 1.0)):
    fig = plt.figure(figsize=figsize)
    fig.text(0.02, 0.5, f"${tex}$", fontsize=fontsize, va="center")
    plt.axis("off")
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

render_eq("eq_tfidf.png",
    r"\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\cdot\log\!\left(\frac{N}{1+\mathrm{df}(t)}\right)")
render_eq("eq_cosine.png",
    r"\mathrm{sim}(q,d) = \frac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\| \, \|\mathbf{d}\|}")
render_eq("eq_softmax.png",
    r"P(y=k\mid x) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x})}{\sum_{j=1}^{K}\exp(\mathbf{w}_j^\top \mathbf{x})}")
render_eq("eq_ce_loss.png",
    r"\mathcal{L}_{CE} = -\sum_{i=1}^{N}\sum_{k=1}^{K} y_{i,k}\,\log \hat{y}_{i,k}")
render_eq("eq_lora.png",
    r"W' = W_0 + \Delta W = W_0 + BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r \ll \min(d,k)")
render_eq("eq_rag.png",
    r"P(y\mid x) = \sum_{z\, \in\, \mathrm{top\text{-}}k(x)} P(z\mid x)\, P(y\mid x, z)")
render_eq("eq_rouge.png",
    r"\mathrm{ROUGE\text{-}N} = \frac{\sum_{S \in \mathrm{Ref}} \sum_{\mathrm{gram}_n \in S} \mathrm{Count}_{\mathrm{match}}(\mathrm{gram}_n)}{\sum_{S \in \mathrm{Ref}} \sum_{\mathrm{gram}_n \in S} \mathrm{Count}(\mathrm{gram}_n)}")

print("figures written to", FIG)
