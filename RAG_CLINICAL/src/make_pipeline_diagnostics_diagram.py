import os
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")

def generate_pipeline_diagnostics(output_path=os.path.join(FIG, "fig_pipeline_diagnostics.png")):
    # Ensure parent directory exists for custom paths as well
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Set professional styling matching the paper's figures
    sns.set_theme(style='whitegrid', font='DejaVu Sans')
    
    # Setup side-by-side subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.9, 6.04), dpi=150)
    
    # Defined corporate/academic color palette
    primary_blue = '#2b5c8f'
    highlight_red = '#cc0000'
    text_color = '#262626'
    grid_color = '#cccccc'
    
    # ------------------ LEFT SUBPLOT: Retrieval Failures on Anxiety ------------------
    conditions = [
        'Acute Low Back Pain',
        'Asthma',
        'Urinary Tract Infection',
        'Hypertension',
        'Upper Respiratory Infection',
        'Type 2 Diabetes',
        'Generalized Anxiety'
    ]
    precision_values = [100.0, 100.0, 100.0, 100.0, 42.9, 42.9, 0.0]
    
    # Plot horizontal bars
    bars = ax1.barh(conditions, precision_values, color=primary_blue, height=0.55)
    
    # Annotate the values next to each bar
    for bar, val in zip(bars, precision_values):
        width = bar.get_width()
        if val == 0.0:
            # Underline the 0% failure case in red text for visual diagnostic emphasis
            ax1.text(2.0, bar.get_y() + bar.get_height()/2.0, '0.0%', 
                     va='center', ha='left', color=highlight_red, fontweight='bold', fontsize=10)
        else:
            ax1.text(width + 1.5, bar.get_y() + bar.get_height()/2.0, f'{val:.1f}%', 
                     va='center', ha='left', color=text_color, fontweight='bold', fontsize=10)
            
    ax1.set_title('Sparse Retrieval Fails on Anxiety due to Lexical Mismatch', 
                 fontsize=12, fontweight='bold', pad=15, color=text_color)
    ax1.set_xlabel('Oracle Retrieval Precision @ 2 (%)', fontsize=11, fontweight='bold', labelpad=10, color=text_color)
    ax1.set_xlim(0, 110)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.tick_params(axis='both', which='major', labelsize=10, labelcolor=text_color)
    
    # Configure axes, grid, and spines
    ax1.grid(True, linestyle='-', color=grid_color, zorder=0)
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
    
    # ------------------ RIGHT SUBPLOT: SOAP Section Classification ------------------
    categories = ['Assessment', 'Objective', 'Plan', 'Subjective', 'Other']
    f1_scores = [100.0, 100.0, 100.0, 92.0, 88.0]
    
    # Plot vertical bars
    bars2 = ax2.bar(categories, f1_scores, color=primary_blue, width=0.45)
    
    # Annotate values on top of each bar
    for bar, val in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + 1.5, f'{val:.1f}%', 
                 ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=10)
        
    ax2.set_title('Robust SOAP Sentence Classification with Minor Dialogue Overlap', 
                 fontsize=12, fontweight='bold', pad=15, color=text_color)
    ax2.set_ylabel('Approximate F1-Score (%)', fontsize=11, fontweight='bold', labelpad=10, color=text_color)
    ax2.set_ylim(0, 110)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=10, color=text_color)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis='both', which='major', labelsize=10, labelcolor=text_color)
    
    # Configure axes, grid, and spines
    ax2.grid(True, linestyle='-', color=grid_color, zorder=0)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)
    
    # ------------------ FIGURE LEVEL LAYOUT ------------------
    fig.suptitle('Pipeline Diagnostics: Section Classification and Retrieval Vulnerabilities', 
                 fontsize=15, fontweight='bold', y=0.98, color=text_color)
    
    
    # Adjust margins to prevent overlapping
    plt.tight_layout(rect=[0, 0.04, 1, 0.93], pad=2.0)
    
    # Save the output image
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"pipeline diagnostics saved")

if __name__ == '__main__':
    generate_pipeline_diagnostics()
