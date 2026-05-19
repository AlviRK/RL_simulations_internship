import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import save_dual_plot
import pandas as pd


#----------------------------
# 1. Insight dynamics (Mirco)
#----------------------------


def plot_insight_dynamics_micro (all_seeds, exp):
        print("     [2/4] Analyzing Dynamics")

        # 1. Selection: 
        session_data = all_seeds[0]  # Pick one Seed for Micro-Plot
        replays = session_data['replays']
        seed_id = replays[0].get('seed', 'N/A')
        total_eps = len(replays)
        indices = {
            "a)": 0,                
            "b)": 49,   # before: total_eps // 2,     
            "c)": 99    # before: total_eps - 1      
        }
        # indices = [0, total_eps //2, total_eps-1] # Catch Episodes 1, 50 and 100 (if 100 total eps)

        event_colors = {
            "FIRST_DIRT": "green", 
            "SECOND_DIRT": "green", 
            "FIRST_VASE": "red", 
            "EXIT": "orange"
        }

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

        for phase_name, idx in indices.items():
            r = replays[idx]
            
            # We create a brand new figure for EACH one
            fig, ax = plt.subplots(figsize=(10, 4))

            # --- DATA EXTRACTION --- 
            signal = np.array(r.get('insight', []))
            rewards = np.array(r.get('rewards', []))
            events = r.get('events', [])
            steps = np.arange(len(signal))              
            cum_reward = np.cumsum(rewards)

            # --- PLOTTING ---
            # A) Primary Axis: Insight Magnitude per step
            ax.plot(steps, signal, label="Insight", color='blue', linewidth=1.5)
            ax.set_ylabel("Insight Magnitude (step)", color='blue')
            ax.set_xlabel("Steps")
            
            # Draw insight event markers
            nonzero_mask = signal != 0
            if np.any(nonzero_mask):
                ax.scatter(steps[nonzero_mask], signal[nonzero_mask], 
                        s=40, marker='o', color='blue', zorder=5, label='Insight Event')
            
            # B) Secondary Axis: Cumulative Reward
            ax2 = ax.twinx()
            ax2.plot(steps, cum_reward, label='Cum. Reward', color='orange', alpha=0.6)
            ax2.set_ylabel("Cumulative Reward (episode)", color='orange')

            # C) Event Markers (Vertical Lines)
            for t, event_list in enumerate(events):
                for ev in event_list:
                    if ev in event_colors:
                        ax.axvline(x=t, color=event_colors[ev], linestyle='--', alpha=0.7)
                        ax.text(t, ax.get_ylim()[1], ev, color=event_colors[ev], 
                                fontsize=8, rotation=90, va='bottom')

            ax.set_title(f"{phase_name} | Episode {r['episode']} | Seed {seed_id}")
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            
            # Save as individual files (e.g., 03_insight_micro_early.png)
            filename = f"03_insight_micro_{phase_name}.png"
            save_dual_plot(filename, "descriptive", exp['id'])
            plt.close()

        # --- PART B: The "Fourth" Figure (The Stacked Multipanel) ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 12)) # Increased height for text labels
        
        for i, (phase_label, idx) in enumerate(indices.items()):
            ax = axes[i]
            r = replays[idx]
            
            signal = np.array(r.get('insight', []))
            rewards = np.array(r.get('rewards', []))
            events = r.get('events', [])
            steps = np.arange(len(signal))              
            cum_reward = np.cumsum(rewards)

            # 1. Insight Line + Scatter (Blue Dots)
            ax.plot(steps, signal, color='blue', linewidth=1.5)
            nonzero_mask = signal != 0
            if np.any(nonzero_mask):
                ax.scatter(steps[nonzero_mask], signal[nonzero_mask], s=40, marker='o', color='blue', zorder=5)

            # 2. Cumulative Reward
            ax2 = ax.twinx()
            ax2.plot(steps, cum_reward, color='orange', alpha=0.6)
            
            # 3. Title with bold label
            ax.set_title(fr"$\mathbf{{{phase_label}}}$ Episode {r['episode']} | Seed {seed_id}", fontsize=12)
            
            # 4. Event Markers (Lines + TEXT)
            for t, event_list in enumerate(events):
                for ev in event_list:
                    if ev in event_colors:
                        ax.axvline(x=t, color=event_colors[ev], linestyle='--', alpha=0.7)
                        ax.text(t, ax.get_ylim()[1], ev, color=event_colors[ev], 
                                fontsize=8, rotation=90, va='bottom')

            # 5. Formatting identical to individual
            ax.set_ylabel("Insight Germain", color='blue')
            ax2.set_ylabel("Cum. Reward", color='orange')
            ax.set_xlabel("Steps")
            ax.grid(True, alpha=0.2)
            
        plt.tight_layout()
        save_dual_plot("03_insight_micro_STACKED_MASTER.png", "descriptive", exp['id'], vector=True)
        plt.close()



def plot_insight_trend(all_seeds, exp):
    # 1. Collect everything into a "Long-form" DataFrame
    episode_data = []
    for session_data in all_seeds:
        replays = session_data['replays']
        seed_id = replays[0].get('seed', 'unknown')
        for r in replays:
            signal = np.array(r.get('insight', []))
            if len(signal) > 0:
                episode_data.append({
                    "Episode": int(r['episode']),
                    # We remove MeanInsight as it dilutes the learning story
                    "MaxInsight": np.max(np.abs(signal)),
                    "Return": r.get('summary', {}).get('total_reward', 0),
                    "Seed": seed_id
                })
    df = pd.DataFrame(episode_data)

    # Increase font size for paper readability
    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Axis 1: Max Insight (The Primary Signal) ---
    # We remove the 'sd' error bar if it's too messy and use 'ci=95' or None.
    # Plotting Max Insight shows the 'surprise peaks' your supervisor requested.
    sns.lineplot(data=df, x="Episode", y="MaxInsight", color='orange', 
                 linewidth=2, label='Peak Insight (±CI 95%)', ax=ax1, 
                 errorbar=('ci', 95), legend=False)
    
    ax1.set_xlabel("Episode", fontweight='bold')
    ax1.set_ylabel("Maximum Episodic Insight (95% CI)", color='orange', fontweight='bold', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='orange')

    # --- Axis 2: Returns (The Performance Metric) ---
    ax2 = ax1.twinx()
    # For return, CI is very important to show stability across seeds.
    sns.lineplot(data=df, x="Episode", y="Return", color='darkgreen', 
                 linewidth=2, label='Mean Episode Return (±95% CI)', ax=ax2, 
                 errorbar=('ci', 95), legend=False)
    
    ax2.set_ylabel("Episode Return", color='darkgreen', fontweight='bold', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    
    # Clean Legend Strategy: Use color-coded axis titles instead of a box if space is tight.
    # Otherwise, merge them clearly:
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines + lines2, labels + labels2, loc="lower center", 
    #           bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    plt.title(f"Insight Decay vs. Performance Convergence", pad=20, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    
    save_dual_plot("04_max_insight_trend.png", "descriptive", exp['id'], dpi=600, vector=True)
    plt.close()



#-------------------------
# 3. Full Insight History
#-------------------------

def plot_full_insight_history(all_seeds, exp):

    # 1. Selection: Use the first seed as the case study
    session_data = all_seeds[0]
    replays = session_data['replays']
    seed_id = replays[0].get('seed', 'N/A')

    full_signal = []
    full_rewards = []
    episode_boundaries = []
    current_step = 0

    for r in replays:
        signal = r.get('insight', [])
        full_signal.extend(np.abs(signal))

        rewards = r.get('rewards', [])
        full_rewards.extend(rewards)

        current_step += len(signal)
        episode_boundaries.append(current_step)

    # 2. Moving Average Calculation
    window_size = 500
    if len(full_rewards) > window_size:
        kernel = np.ones(window_size) / window_size
        moving_average = np.convolve(full_rewards, kernel, mode='valid')
    # Step B: Pad the tail so the length matches full_signal
        # This extends the final average value to the end of the plot
        padding_length = len(full_signal) - len(moving_average)
        if padding_length > 0:
            final_val = moving_average[-1]
            padding = np.full(padding_length, final_val)
            moving_average = np.concatenate([moving_average, padding])
    else:
        moving_average = np.zeros(len(full_rewards))

    # 3. Visualization
    #plt.rcParams.update({'font.size': 25})
    fig, ax1 = plt.subplots(figsize=(30, 10))
    
    steps = np.arange(len(full_signal)) # Ensure X-axis alignment
    
    # Insight Traces
    l1 = ax1.plot(steps, full_signal, color='blue', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Total Lifetime Steps", fontsize=18)
    ax1.set_ylabel("Insight Magnitude", color='blue', fontweight='bold', fontsize=18)
    
    # Reward Density
    ax2 = ax1.twinx()
    # Now moving_average is the same length as steps
    l2 = ax2.plot(steps, moving_average, color='seagreen', linewidth=1.2, alpha=0.9)
    ax2.set_ylabel(f"Reward Density ({window_size}-step MA | terminal values padded", 
                   color='seagreen', fontweight='bold', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='seagreen')

    # Episode Boundaries
    # (Only draw every 5th or 10th if it's too crowded)
    for i, boundary in enumerate(episode_boundaries[:-1]):
        if i % 2 == 0: # Lighten the vertical lines
            ax1.axvline(boundary, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)

    # Legends & Titles
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs, loc="upper center", bbox_to_anchor=(0.5, 1.15), 
    #           ncol=2, frameon=False)

    plt.title(f"Continuous Insight History (Seed: {seed_id})", pad=20, fontsize=18)
    
    save_dual_plot("04b_full_insight_history.png", "descriptive", exp['id'], dpi=600, vector=True)
    plt.close()




#-----------------------
# 4. Calculate Heatmaps
#-----------------------
def plot_spatial_heatmaps(all_seeds, exp, env_meta):
    print("     [2/4] Analyzing Dynamics (Heatmaps)")

    # # 1. Filter for Seed 42 only
    # seed_42_data = [s for s in all_seeds if s.get('agent', {}).get('settings', {}).get('seed') == 42]
    
    # if not seed_42_data:
    #     print("⚠️ Seed 42 not found in the provided data!")
    #     return

    # # 2. Setup Grid
    # w, h = env_meta.get('num_cols'), env_meta.get('num_rows')
    # total_cells = w * h
    # replays = seed_42_data[0]['replays']
    # replays.sort(key=lambda x: x['episode'])
    
    # 1. Setup Grid
    w, h = env_meta.get('num_cols'), env_meta.get('num_rows')
    total_cells = w * h

    # 2. Flatten all seeds into one big list of replays
    replays = []
    for session_data in all_seeds:
        replays.extend(session_data['replays'])
    replays.sort(key=lambda x: x['episode'])

    # 3. Define Phases
    total = len(replays)
    phases = {
        "a) Early": replays[:total//3],
        "b) Middle": replays[total//3 : 2*total//3],
        "c) Late": replays[2*total//3:]
    }

    # 4. Calculate heatmaps
    maps = {}                                            
    global_max = 0                                       

    for name, phase_data in phases.items():
        insight_grid = np.zeros((h, w))
        visit_grid = np.zeros((h, w))

        for r in phase_data:
            states = r.get('states', [])
            signal = r.get('insight', [])
            length = min(len(states), len(signal))

            for t in range(length):
                cell_id = states[t] % total_cells
                r_idx, c_idx = int(cell_id // w), int(cell_id % w)

                if 0 <= r_idx < h and 0 <= c_idx < w:
                    insight_grid[r_idx, c_idx] += abs(signal[t])
                    visit_grid[r_idx, c_idx] += 1

        with np.errstate(invalid='ignore'):
            mean_map = np.nan_to_num(np.divide(insight_grid, visit_grid))

        maps[name] = mean_map
        global_max = max(global_max, np.max(mean_map))

    # 5. Plotting each heatmap in its OWN separate file!
    for name, heatmap_data in maps.items():
        # Create a square plot for the single grid
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # We still use global_max so the colors mean the same thing across all 3 images!
        sns.heatmap(heatmap_data, ax=ax, cmap="magma", vmin=0, vmax=global_max, square=True)
        
        ax.set_title(f"{name} Phase - Mean Insight (All Seeds)")
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save as individual files (e.g., 05_heatmap_Early.png)
        filename = f"05_heatmap_{name}.png"
        save_dual_plot(filename, "descriptive", exp['id'])
        plt.close()


    # Plot all three in ONE combined 1x3 figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})
    
    sorted_names = sorted(maps.keys())
    im = None # We'll use this to link the colorbar

    for i, name in enumerate(sorted_names):
        ax = axes[i]
        
        # Plot WITHOUT the built-in seaborn colorbar
        sns.heatmap(maps[name], ax=ax, cmap="magma", vmin=0, vmax=global_max, 
                    square=True, cbar=False)
        
        # Store the mapping from one of the plots to use for the global colorbar
        if i == 0:
            im = ax.collections[0]
            
        ax.set_title(f"{name} Phase", fontsize=14, fontweight='bold')
        ax.axis('off')

    # Now add one GLOBAL colorbar that doesn't push the plots around
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6]) 
    fig.colorbar(im, cax=cbar_ax, label='Mean Insight')

    # Adjust layout to make sure the colorbar doesn't overlap
    plt.subplots_adjust(right=0.9, wspace=0.1) 
    
    save_dual_plot("05_spatial_heatmap_combined.png", "descriptive", exp['id'])
    plt.close()

