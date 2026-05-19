import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. GLOBAL SETTINGS & STYLE
# ==========================================
# Paths relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
BASE_LOG_DIR = CURRENT_DIR.parents[2] / "01_code" / "src" / "replay" / "logs" # Location of JSON file
CODE_OUT_DIR = CURRENT_DIR.parent / "output" / "demos" / "190526" # Location of Sim Results

PROJECT_ROOT = CURRENT_DIR.parents[3] # up 5 levels to the Project Root
#PAPER_OUT_DIR = PROJECT_ROOT / "05_manuscripts/insight/01_paper/00_demo/08_190526/outputs/"


def save_dual_plot(filename, subfolder, experiment_id, vector=False, dpi=300):
    """
    Saves the current matplotlib figure to BOTH:
    1. The Code Output folder
    2. The Paper Output folder
    """
    # Define the two destination folders
    dest_code = CODE_OUT_DIR / experiment_id / subfolder
    #dest_code = CODE_OUT_DIR
    #dest_paper = PAPER_OUT_DIR / experiment_id / subfolder
    
    # Ensure they exist
    dest_code.mkdir(parents=True, exist_ok=True)
    #dest_paper.mkdir(parents=True, exist_ok=True)
    
    # Save to both
    ## 1. PNG
    plt.savefig(dest_code / filename, dpi=dpi, bbox_inches='tight')
    #plt.savefig(dest_paper / filename, dpi=300, bbox_inches='tight')

    ## 2. SVG
    if vector:
        svg_filename = Path(filename).with_suffix('.svg')
        plt.savefig(dest_code / svg_filename, format='svg', bbox_inches='tight')

    # Save PDF (Standard LaTeX Vector Format)
    pdf_filename = Path(filename).with_suffix('.pdf')
    plt.savefig(dest_code / pdf_filename, format='pdf', bbox_inches='tight', transparent=True)
    #plt.savefig(dest_paper / pdf_filename, format='pdf', bbox_inches='tight', transparent=True)
    
    print(f"   -> Saved: {subfolder}/{filename}")