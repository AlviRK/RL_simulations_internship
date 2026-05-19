import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy import stats

from utils import save_dual_plot, BASE_LOG_DIR

import baseline_analysis as ba
import dynamics_analysis as da
import event_analysis as ea
import validity_analysis as va




# Plot Styling
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")



# ==========================================
# 2. EXPERIMENT CONFIGURATION
# ==========================================
# This is the "Control Center". 
# For now, it has 1 experiment.

EXPERIMENTS = [ 
    {
        "id": "td_room1",    # Used for Output Folder Name
        "title": "TD (Room1)",   # Used for Plot Titles
        "filename": "training_data.json",  # File that is loaded
        #"filename": "sandbox_room1_1.json",
        "signal": "Insight",
        "signal_version": "v01",
        "agent_type": "TD",
        # TODO: Add different variables or dicts for hyperparameters and Insight versions etc.
    }
]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_input_path(experiment):  # Calculates input/output paths based on experiment config
    # 1. Check for organized structure: logs/td_room1/replays.json
    #organized_path = BASE_LOG_DIR / experiment["id"] / experiment["filename"]
    organized_path = BASE_LOG_DIR
    if organized_path.exists():
        return organized_path
        
    # 2. Check for flat structure: logs/replays.json (Current setup)
    flat_path = BASE_LOG_DIR / experiment["filename"]
    if flat_path.exists():
        return flat_path
        
    return None # File not found



def load_all_seeds(experiment):
    """
    Finds the experiment folder and loads EVERY seed_*.json file found inside.
    """
    print(f"Looking for seed files in: {BASE_LOG_DIR}")
    
    # Grab all .json files in that directory
    json_files = sorted(list(BASE_LOG_DIR.glob("seed_*.json")))
    
    if not json_files:
        print(f"⚠️  CRITICAL: No files matching 'seed_*.json' found in {BASE_LOG_DIR}!")
        return None

    all_loaded_data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if 'replays' in data:
                # Extract the seed listed in the JSON file
                actual_seed = data.get('agent', {}).get('settings', {}).get('seed', 'unknown')
                for r in data['replays']:
                    r['seed'] = actual_seed
            
            all_loaded_data.append(data)
            print(f"✅ Loaded: {file_path.name}")
            
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            
    print(f"🎯 Successfully loaded {len(all_loaded_data)} total seeds.")
    return all_loaded_data


def main():
    print("--- Starting Analysis ---")

    for exp in EXPERIMENTS:
        print(f"\nProcessing Experiment: {exp['title']}...")

        all_seeds = load_all_seeds(exp) 
        if not all_seeds:
            continue

        env_meta = all_seeds[0].get('env_meta', {})

        # Part 1: Baseline (Learning Curves)
        ba.plot_baseline(all_seeds, exp)

        # Part 2: Dynamics (Traces & Heatmaps)
        da.plot_insight_dynamics_micro(all_seeds, exp)
        da.plot_insight_trend(all_seeds, exp)
        da.plot_full_insight_history(all_seeds, exp)
        da.plot_spatial_heatmaps(all_seeds, exp, env_meta)

        # # Part 3: Event Analysis (Contrast & Discovery)
        ea.plot_event_contrasts(all_seeds, exp)
        ea.plot_first_discovery(all_seeds, exp)

        # # Part 4: Validity Analysis (Habituation, Reward, Policy, Predictive)
        va.plot_visitation_decay(all_seeds, exp, env_meta)
        va.plot_reward_proportionality(all_seeds, exp)
        va.plot_policy_change_relevance(all_seeds, exp)
        va.plot_predictive_validity(all_seeds, exp)
        

        
    print("\n--- Analysis finished ---")
          

if __name__ == "__main__":
    main()