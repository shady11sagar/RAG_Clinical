import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.5)
ax.axis("off")

W, H, GAP = 2.3, 1.3, 0.35
COL1, COL2, COL3, COL4 = 0.4, 0.4 + W + GAP, 0.4 + 2*(W+GAP), 0.4 + 3*(W+GAP)
ROW_TOP, ROW_BOT = 4.3, 1.0
WIDE_W = 2 * W + GAP  # merged col3+col4 width for the bottom-right box

boxes = [
    (COL1, ROW_TOP, W, H, "Raw Clinical Input\n(EHR text / ambient\ndialogue audio)", "#dde6f2"),
    (COL2, ROW_TOP, W, H, "Phase I\nSecure De-identification\n(regex + clinical NER)", "#c3d5ea"),
    (COL3, ROW_TOP, W, H, "Phase II\nDomain-Adapted Generator\n(LoRA/QLoRA LLM)\n→ Draft S/O", "#a9c4e0"),
    (COL4, ROW_TOP, W, H, "Phase III\nRetrieval Layer\n(vector DB of guidelines)\n→ Grounded A/P", "#8fb3d6"),

    (COL1, ROW_BOT, W, H, "Clinician Review\n& Sign-off\n(human-in-the-loop)", "#f2e2c4"),
    (COL2, ROW_BOT, W, H, "Phase V\nEvaluation & Security Audit\n(ROUGE, BERTScore,\nred-team suite)", "#c3d5ea"),
    (COL3, ROW_BOT, WIDE_W, H, "Assembled SOAP Note\n(S/O + grounded Assessment/Plan, source-cited)", "#dde6f2"),
]

for x, y, w, h, text, color in boxes:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                          linewidth=1.2, edgecolor="#1f3864", facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.3, color="#1a1a1a")

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.4, color="#1f3864", shrinkA=0, shrinkB=0)
    ax.add_patch(a)

mid_top = ROW_TOP + H/2
mid_bot = ROW_BOT + H/2

# Top row: strictly horizontal, left to right
arrow(COL1 + W, mid_top, COL2, mid_top)
arrow(COL2 + W, mid_top, COL3, mid_top)
arrow(COL3 + W, mid_top, COL4, mid_top)


arrow(COL3 + W/2, ROW_TOP, COL3 + W/2, ROW_BOT + H)
arrow(COL4 + W/2, ROW_TOP, COL4 + W/2, ROW_BOT + H)

# Bottom row: strictly horizontal, right to left (the "return" of the snake)
arrow(COL3, mid_bot, COL2 + W, mid_bot)
arrow(COL2, mid_bot, COL1 + W, mid_bot)

# Phase IV annotation (offset, non-overlapping)
ax.annotate("Phase IV: Ambient Clinical Intelligence\n(speech-to-text feeds raw input; structured\noutput returns to clinician workflow)",
            xy=(COL1 + 0.3, ROW_TOP), xytext=(COL1, ROW_TOP - 1.6), fontsize=8.5, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))


fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_architecture.png"), dpi=220)
plt.close(fig)
print("architecture diagram saved")