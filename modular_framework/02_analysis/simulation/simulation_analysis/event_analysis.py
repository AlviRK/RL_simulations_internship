import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_dual_plot





#--------------------
# 1. Event Contrasts
#--------------------

def plot_event_contrasts(all_seeds, exp):
    print("     [3/4] Analyzing Events")

    records = []

    # 1. Loop through every seed
    for seed_data in all_seeds:
        replays = seed_data['replays']

        # Sort here to ensure phases are assigned correctly
        replays.sort(key=lambda x: x['episode'])

        total_episodes = len(replays)
        seed_id = replays[0].get('seed', 'unknown')

        for r in replays:
            ep_idx = r['episode']

            # Determine phase
            if ep_idx < total_episodes / 3:
                phase = "Early"
            elif ep_idx < 2 * total_episodes / 3:
                phase = "Middle"
            else:
                phase = "Late"

            signal = np.array(r.get('insight', []))
            rewards = r.get('rewards', [])

            # Identify event types for every occurrence
            for t, reward in enumerate(rewards):
                if t >= len(signal): break

                if reward == 0.5:
                    event_type = "Dirt"
                elif reward == -1.0:
                    event_type = "Vase"
                elif reward == 1.0:
                    event_type = "Exit"
                else:
                    event_type = "Baseline"

                # Record the Signal
                val = abs(signal[t])

                # Store Data for Statistical Analysis
                records.append({
                    "Seed": seed_id,
                    "Episode": ep_idx,
                    "Phase": phase,
                    "Type": event_type,
                    "Insight": val
                })

    # 2. Create Master DataFrame
    df = pd.DataFrame(records)


    # --- CHECK 2: DATA INTEGRITY AUDIT ---
    print("\n--- DATA AUDIT ---")
    # Verify counts for every combination of Phase and Event Type
    counts = df.groupby(['Phase', 'Type']).size().unstack(fill_value=0)
    print("Event Counts per Phase:\n", counts)
    
    # Check for NaNs that might suppress plotting
    nan_count = df['Insight'].isna().sum()
    if nan_count > 0:
        print(f"CRITICAL: Found {nan_count} NaN values in Insight signal.")
    
    # Check if 'Middle' vases have zero magnitude vs just being absent
    if "Middle" in df['Phase'].values:
        mid_vases = df[(df['Phase'] == 'Middle') & (df['Type'] == 'Vase')]
        if not mid_vases.empty:
            print(f"Middle Vase Mean Magnitude: {mid_vases['Insight'].mean()}")
        else:
            print("ALERT: No Vase hits recorded in the Middle Phase.")
    print("------------------\n")

    # -------- AUDIT OVER ------------

    type_order = ["Baseline", "Dirt", "Vase", "Exit"]
    phase_order = ["Early", "Middle", "Late"]

    # --- NEW: HIERARCHICAL AGGREGATION ---
    # This calculates the average insight for each Event Type within each Seed.
    # It prevents "watering down" the signal with high-frequency baseline steps.
    df_seed_overall = df.groupby(['Seed', 'Type'])['Insight'].mean().reset_index()

    # This does the same but keeps the Phase (Early, Middle, Late) intact.
    df_seed_phase = df.groupby(['Seed', 'Phase', 'Type'])['Insight'].mean().reset_index()
    # -------------------------------------

    # --- PLOT 1: Overall Aggregated Contrast ---
    plt.figure(figsize=(8, 6))
    # CHANGE: Use 'df_seed_overall' instead of 'df'
    sns.barplot(data=df_seed_overall, x="Type", y="Insight", order=type_order, 
                palette="viridis", errorbar=('ci', 95), capsize=.1)
    # ... (rest of your titles/labels)
    
    plt.title(f"Event Responsiveness\n(Aggregated: {len(all_seeds)} Seeds, all episodes)")
    plt.ylabel("Mean Insight Magnitude (±95% CI)")
    plt.xlabel("Event Type")
    save_dual_plot("06_event_contrast_overall.png", "evaluative", exp['id'])
    plt.close()

    # --- PLOT 2: Evolution by Phase (Aggregated) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Phase", y="Insight", hue="Type", 
                order=phase_order, hue_order=type_order,
                palette="magma", errorbar=('ci', 95), capsize=.05)
    
    plt.title(f"{exp['title']} - Signal Habituation by Learning Phase\n(Mean Response Aggregated over {len(all_seeds)} Seeds, all episodes)")
    plt.ylabel("Mean Insight Magnitude (±95% CI)")
    plt.legend(title="Event Type", loc='upper right')
    
    save_dual_plot("07_event_contrast_evolution.png", "evaluative", exp['id'])
    plt.close()




#---------------------------------------
# 2. First vs. later discovery contrasts
#---------------------------------------
def plot_first_discovery(all_seeds, exp):
    records = []

    for seed_data in all_seeds:
        replays = seed_data['replays']
        seed_id = replays[0].get('seed', 'unknown')
        
        # Track counts per event type FOR THIS SEED
        counts = {"Dirt": 0, "Vase": 0, "Exit": 0}
        
        # Ensure chronological order
        sorted_replays = sorted(replays, key=lambda x: x['episode'])

        for r in sorted_replays:
            signal = np.array(r.get('insight', []))
            rewards = r.get('rewards', [])

            for t, reward in enumerate(rewards):
                if t >= len(signal): break

                # 1. Identify Event Type
                event_type = None
                if reward == 0.5: event_type = "Dirt"
                elif reward == -1.0: event_type = "Vase"
                elif reward == 1.0: event_type = "Exit"

                if event_type:
                    # 2. Increment occurrence count
                    counts[event_type] += 1
                    
                    # 3. Single-Point Sampling (No window_k)
                    val = abs(signal[t])

                    records.append({
                        "Seed": seed_id,
                        "Type": event_type,
                        "Occurrence": counts[event_type],
                        "Insight": val
                    })

    df = pd.DataFrame(records)

    # 4. Define Bins and Labels
    # Use right=True (default) so (0, 1] is '1', (1, 2] is '2', etc.
    bins = [0, 1, 2, 5, 10, 20, 50, float('inf')]
    labels = ["1", "2", "3-5", "6-10", "11-20", "21-50", ">50"]
    df['Bin'] = pd.cut(df['Occurrence'], bins=bins, labels=labels)

    # --- DATA AUDIT ---
    print("\n--- HABITUATION AUDIT ---")
    print(df.groupby(['Type', 'Bin'], observed=False).size().unstack(fill_value=0))
    print("-------------------------\n")

    # --- PLOT: Habituation Curve (Line Plot) ---
    plt.figure(figsize=(10, 6))
    # Lineplot automatically calculates Mean + 95% CI shadow across seeds
    sns.lineplot(data=df, x="Bin", y="Insight", hue="Type", marker="o",
                 hue_order=["Dirt", "Vase", "Exit"], palette="viridis")
    
    plt.title(f"{exp['title']} - Insight Habituation Curve")
    plt.ylabel("Insight Magnitude (at t=0)")
    plt.xlabel("Occurrence Number")
    plt.grid(True, alpha=0.3)
    
    save_dual_plot("08_habituation_curve.png", "evaluative", exp['id'])
    plt.close()

    # --- PLOT: Binned Distribution (Box Plot) ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Bin", y="Insight", hue="Type", 
                hue_order=["Dirt", "Vase", "Exit"], showfliers=False)
    
    plt.title(f"{exp['title']} - Signal Variance by Experience")
    save_dual_plot("09_habituation_distribution.png", "evaluative", exp['id'])
    plt.close()
