import os
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")

def generate_rouge_groundedness_tradeoff(output_path=os.path.join(FIG, "fig_rouge_groundedness_tradeoff.png")):
    # Ensure parent directory exists for custom paths as well
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    sns.set_theme(style='whitegrid', font='DejaVu Sans')
    
    # Setup side-by-side subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.83, 6.04), dpi=150)
    
    color_no_rag = '#5975a4'    # Slate blue
    color_rag_aug = '#cc8963'    # Warm clay/brown
    text_color = '#262626'
    grid_color = '#cccccc'
    
    # ------------------ LEFT SUBPLOT: Lexical Text Similarity (ROUGE) ------------------
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    no_rag_scores = [0.679, 0.640, 0.673]
    rag_aug_scores = [0.663, 0.570, 0.619]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot grouped bars
    rects1 = ax1.bar(x - width/2, no_rag_scores, width, label='No-RAG Baseline', color=color_no_rag)
    rects2 = ax1.bar(x + width/2, rag_aug_scores, width, label='RAG-Augmented', color=color_rag_aug)
    
    # Add numerical labels on top of each bar
    for rect in rects1:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, height + 0.015, f'{height:.3f}', 
                 ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=10)
                 
    for rect in rects2:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, height + 0.015, f'{height:.3f}', 
                 ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=10)
        
    ax1.set_title('Lexical Text Similarity (ROUGE Metrics)', 
                 fontsize=12, fontweight='bold', pad=15, color=text_color)
    ax1.set_ylabel('F-Measure Score (0.0 to 1.0)', fontsize=11, fontweight='bold', labelpad=10, color=text_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, color=text_color)
    ax1.set_ylim(0.0, 1.1)
    ax1.tick_params(axis='y', which='major', labelsize=10, labelcolor=text_color)
    
    # Configure axes, grid, and spines
    ax1.grid(True, linestyle='-', color=grid_color, zorder=0)
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
    

    legend = ax1.legend(title='Configuration', loc='lower left', frameon=True)
    legend.get_title().set_fontweight('bold')
    
    # ------------------ RIGHT SUBPLOT: Guideline Groundedness ------------------
    categories = ['No-RAG Baseline', 'RAG-Augmented']
    groundedness_scores = [0.112, 0.949]
    
    # Plot vertical bars with distinct colors for each category
    bars2 = ax2.bar(categories, groundedness_scores, width=0.5, color=[color_no_rag, color_rag_aug])
    
    # Add numerical labels on top of each bar
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + 0.015, f'{height:.3f}', 
                 ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=10)
        
    ax2.set_title('Guideline Groundedness (Plan Cosine Sim.)', 
                 fontsize=12, fontweight='bold', pad=15, color=text_color)
    ax2.set_ylabel('Mean Cosine Similarity', fontsize=11, fontweight='bold', labelpad=10, color=text_color)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(categories, fontsize=10, color=text_color)
    ax2.set_ylim(0.0, 1.1)
    ax2.tick_params(axis='y', which='major', labelsize=10, labelcolor=text_color)
    
    # Configure axes, grid, and spines
    ax2.grid(True, linestyle='-', color=grid_color, zorder=0)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)
    
    # ------------------ FIGURE LEVEL LAYOUT ------------------
    fig.suptitle('RAG Grounding Boosts Guideline Alignment 8.5x (0.112 to 0.949) with Minimal ROUGE Decay (<11%)', 
                 fontsize=14, fontweight='bold', y=0.98, color=text_color)
    

    # Adjust margins to prevent overlapping
    plt.tight_layout(rect=[0, 0.04, 1, 0.93], pad=2.0)
    
    # Save the output image
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"rouge groundedness tradeoff saved")

if __name__ == '__main__':
    generate_rouge_groundedness_tradeoff()
