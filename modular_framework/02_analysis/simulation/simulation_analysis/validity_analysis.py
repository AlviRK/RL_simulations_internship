import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils import save_dual_plot



#---------------------------------------
# 1. Habituation/Visitation Decay
#---------------------------------------

def plot_visitation_decay(all_seeds, exp, env_meta):
    print("     [4/4] Analyzing Validity")
    
    # 1. Parse Map Metadata
    dirt_cells = set(env_meta.get('dirt_cells', []))
    vase_cells = set(env_meta.get('vase_cells', []))
    terminal_cell = env_meta.get('terminal_cell', -1)
    num_cells = env_meta.get('num_cells', 100)
    
    records = []
    
    for seed_data in all_seeds:
        replays = seed_data['replays']
        seed_id = replays[0].get('seed', 'unknown')
        state_counts = {} 
        
        # Sort chronologically
        sorted_replays = sorted(replays, key=lambda x: x['episode'])
        
        for r in sorted_replays:
            states = r.get('states', [])
            signal = np.array(r.get('insight', []), dtype=float)
            rewards = r.get('rewards', [])

            for t, reward in enumerate(rewards):
                if t >= len(signal): break

                if reward == 0.5:
                    event_type = "Dirt"
                elif reward == -1.0:
                    event_type = "Vase"
                elif reward == 1.0:
                    event_type = "Exit"
                else:
                    event_type = "Floor"

                val = abs(signal[t])
                s_t = states[t]

                state_counts[s_t] = state_counts.get(s_t, 0) + 1
                n_st = state_counts[s_t]

                
                records.append({
                    "Seed": seed_id, "Insight": val,
                    "Visits": n_st, "Type": event_type
                })
            
    df = pd.DataFrame(records)
    
    # 3. Binning
    bins = [0, 1, 2, 5, 10, 20, 50, float('inf')]
    labels = ["1", "2", "3-5", "6-10", "11-20", "21-50", ">50"]
    df['VisitBin'] = pd.cut(df['Visits'], bins=bins, labels=labels)
    
    # --- PLOT A: Aggregated Overall Decay ---
    plt.figure(figsize=(10, 6))
    
    # Log-correlation across all seeds
    corr_log = np.corrcoef(np.log1p(df['Visits']), df['Insight'])[0,1]
    
    sns.pointplot(data=df, x="VisitBin", y="Insight", 
                  errorbar=('ci', 95), capsize=0.1, 
                  color="teal", markers="o", linestyles="-")
    
    plt.title(f"{exp['title']} - Aggregated Habituation Dynamics ({len(all_seeds)} Seeds)")
    plt.xlabel("Number of Times State Visited")
    plt.ylabel("Mean Insight Magnitude")
    plt.text(0.95, 0.95, f"Avg Log-Correlation: {corr_log:.2f}", 
             transform=plt.gca().transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    save_dual_plot("10a_visitation_decay_overall.png", "validity", exp['id'])
    plt.close()

    # --- PLOT B: Aggregated Split by Type ---
    plt.figure(figsize=(10, 6))
    type_palette = {"Dirt": "green", "Vase": "red", "Exit": "orange", "Floor": "gray"}
    
    sns.pointplot(data=df, x="VisitBin", y="Insight", hue="Type",
                  palette=type_palette,
                  hue_order=["Dirt", "Vase", "Exit", "Floor"],
                  errorbar=('ci', 95), capsize=0.1)
    
    plt.title(f"{exp['title']} - Aggregated Habituation Dynamics per Event ({len(all_seeds)} Seeds)")
    plt.grid(True, alpha=0.3)
    save_dual_plot("10b_visitation_decay_split.png", "validity", exp['id'])
    plt.close()


#---------------------------------------
# 2. Reward Proportioality
#---------------------------------------

def plot_reward_proportionality(all_seeds, exp):

    records = []
    valid_rewards = {-1.0, 0.0, 0.5, 1.0}

    # 1. Loop through every seed
    for seed_data in all_seeds:
        replays = seed_data['replays']
        seed_id = replays[0].get('seed', 'unknown')

        for r in replays:
            rewards = r.get('rewards', [])
            insight = np.array(r.get('insight', []), dtype=float)
            # limit = min(len(rewards), len(insight))
            episode_num = r.get('episode', '??')

            for t, r_t in enumerate(rewards):
                # if t >= len(insight): break
                r_t = float(rewards[t])

                # --- DIAGNOSTIC PRINTS ---
                # if r_t == 1.0:
                #     print(f"[CHECK] Found Exit Reward (1.0) at step {t} in Seed {seed_id}, Ep {episode_num}")

                try:
                    val = abs(insight[t])
                    
                    # Store data exactly as it is (no rounding)
                    records.append({
                        "Reward": r_t,
                        "AbsReward": abs(r_t),
                        "Insight": val,
                        "Episode": int(episode_num)
                    })

                except IndexError:
                    # This triggers if rewards is longer than insight
                    if r_t == 1.0:
                        print(f"  !! MISSING INSIGHT: Step {t} has a reward of 1.0 but no matching insight value.")
                    else:
                        print(f"  !! Index mismatch at step {t}: Reward={r_t}, but no insight exists.")
                
    df = pd.DataFrame(records)

    # Inspect unique reward values to see floating point behavior
    # print("\n[DATA INSPECTION] Unique reward values found in DataFrame:")
    # print(df['Reward'].unique())

    # 2. Calculate Global Correletion
    corr_val = df['AbsReward'].corr(df['Insight'])

    # 3. Visualization
    plt.figure(figsize=(8, 6))

    # # Boxplot shows the distribution across ALL seeds
    sns.boxplot(data=df, x="Reward", y="Insight", 
                order=[-1.0, 0.0, 0.5, 1.0],
                palette="viridis", showfliers=False) 
    
    # Overlay individual points (sampled if data is too huge)
    sample_df = df.sample(n=min(len(df), 2000)) # Limit points to keep PDF size small
    sns.stripplot(data=sample_df, x="Reward", y="Insight", 
                  order=[-1.0, 0.0, 0.5, 1.0],
                  color="black", alpha=0.3, jitter=0.2, size=1.5)
    

    plt.title(f"{exp['title']} - Insight Proportionality (Aggregated)")
    plt.xlabel("Reward Value ($r_t$)")
    plt.ylabel("Insight Magnitude")
    
    # Add stats box
    stats_text = f"Global Correlation ($|R|$ vs Insight): {corr_val:.2f}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             ha='left', va='top', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    save_dual_plot("11_reward_proportionality.png", "validity", exp['id'])
    plt.close()

    prove_trauma_hypothesis(df, exp)



def prove_trauma_hypothesis(df, exp):
    # 1. Check the 'density' of experiences over time
    for r_val in [-1.0, 1.0]:
        subset = df[df['Reward'] == r_val]
        avg_ep = subset['Episode'].mean()
        std_ep = subset['Episode'].std()
        print(f"Reward {r_val}: Mean Episode = {avg_ep:.2f}, Std Dev = {std_ep:.2f}")

    # 2. Plot Insight vs Episode for Vases vs Exits
    plt.figure(figsize=(10, 5))
    for r_val, color, label in [(-1.0, 'red', 'Vase'), (0.5, 'green', 'Dirt'), (1.0, 'orange', 'Exit')]:
        subset = df[df['Reward'] == r_val]
        sns.scatterplot(data=subset, x='Episode', y='Insight', color=color, label=label, alpha=0.5)
    
    plt.title("Trauma Hypothesis: Insight Timing: Vases (Accident) vs Exit (Choice)")
    plt.xlabel("Episode Number")
    plt.ylabel("Insight Magnitude")
    save_dual_plot("11b_trauma_hypothesis.png", "validity", exp['id'])





#---------------------------------------
# 3. Policy Change Relevance
#---------------------------------------


def plot_policy_change_relevance(all_seeds, exp):

    records = []

    # 1. Loop through every seed
    for seed_data in all_seeds:
        replays = seed_data['replays']
        seed_id = replays[0].get('seed', 'unknown')
        
        for r in replays:
            if 'policy_changes' not in r:
                continue
                
            insight = np.array(r.get('insight', []), dtype=float)
            changes = r.get('policy_changes', [])

            for t, val in enumerate(insight):
                records.append({
                    "Seed": seed_id,
                    "Insight": abs(val),
                    "PolicyChange": "Yes" if bool(changes[t]) else "No"
                })
            
    df = pd.DataFrame(records)
    
    if df.empty:
        print("⚠️  WARNING: No 'policy_changes' data found. Skipping.")
        return

    # 2. Global Statistics
    mean_yes = df[df['PolicyChange'] == 'Yes']['Insight'].mean()
    mean_no = df[df['PolicyChange'] == 'No']['Insight'].mean()
    lift = mean_yes / mean_no if mean_no > 0 else 0



    # 3. Visualization
    plt.figure(figsize=(8, 6))
    pal = {"Yes": "#e74c3c", "No": "#34495e"}
    
    # Aggregated Boxplot
    sns.boxplot(data=df, x="PolicyChange", y="Insight", 
                order=["No", "Yes"], palette=pal, showfliers=False)
                
    # Add a point plot to show the mean specifically (often clearer than median in boxes)
    sns.pointplot(data=df, x="PolicyChange", y="Insight", 
                  order=["No", "Yes"], color="purple", markers="D", scale=0.8)

    plt.title(f"{exp['title']} - Policy Change Relevance (Aggregated)")
    plt.xlabel("Policy Change immediately after Insight?")
    plt.ylabel("Insight Magnitude")
    
    # Stats Annotation
    stats_text = f"Mean Lift: {lift:.1f}x\n(Consensus across seeds)"
    plt.text(0.5, 0.9, stats_text, transform=plt.gca().transAxes, 
             ha='center', va='top', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    save_dual_plot("12_policy_change_relevance.png", "validity", exp['id'])
    plt.close()



#---------------------------------------
# 4. Predictive Validity
#---------------------------------------

def plot_predictive_validity(all_seeds, exp):

    max_step_lag = 20
    lags = range(1, max_step_lag + 1)
    step_correlations = []
    
    # Pool data from all seeds
    all_episode_data = []
    for seed_data in all_seeds:
        for r in seed_data['replays']:
            ins = np.abs(np.array(r.get('insight', []), dtype=float))
            rew = np.array(r.get('rewards', []), dtype=float)
            limit = min(len(ins), len(rew))
            if limit > 0:
                all_episode_data.append((ins[:limit], rew[:limit]))

    for k in lags:
        x_vals, y_vals = [], []
        for ins, rew in all_episode_data:
            if len(ins) > k:
                x_vals.extend(ins[:-k])
                y_vals.extend(rew[k:])
        
        corr = np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 10 else 0
        step_correlations.append(corr)

    # --- PART 2: Episode-Level Lag (Macro) ---
    max_ep_lag = 10
    ep_lags = np.array(range(1, max_ep_lag + 1))
    
    # List to store correlation arrays for each seed
    seed_corrs = []
    
    for seed_data in all_seeds:
        sorted_replays = sorted(seed_data['replays'], key=lambda x: x['episode'])
        ep_insights = np.array([np.mean(np.abs(r.get('insight', [0]))) for r in sorted_replays])
        ep_returns = np.array([sum(r.get('rewards', [0])) for r in sorted_replays])
        
        current_seed_lags = []
        for k in ep_lags:
            valid_n = len(ep_insights) - k
            if valid_n < 5:
                current_seed_lags.append(np.nan)
                continue
            
            # Improvement: G_{e+k} - G_{e}
            y_delta = ep_returns[k:] - ep_returns[:valid_n]
            corr = np.corrcoef(ep_insights[:valid_n], y_delta)[0, 1]
            current_seed_lags.append(corr)
        seed_corrs.append(current_seed_lags)

    # Convert to array for Seaborn (Seeds x Lags)
    ep_corr_df = pd.DataFrame(seed_corrs, columns=ep_lags).melt(var_name='Lag', value_name='Correlation')

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Step Lag (Aggregated Mean)
    ax1.plot(lags, step_correlations, marker='o', color='purple', linewidth=2, label='Pooled Correlation')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Micro: Insight$_t$ predicting Reward$_{t+k}$")
    ax1.set_xlabel("Step Delay ($k$)")
    ax1.set_ylabel("Correlation")
    
    # Plot 2: Episode Lag (Mean + CI across Seeds)
    sns.lineplot(data=ep_corr_df, x='Lag', y='Correlation', ax=ax2, 
                 marker='s', color='darkblue', errorbar=('ci', 95))
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Macro: Insight$_e$ predicting Improvement$_{e+k}$")
    ax2.set_xlabel("Episode Delay ($k$)")
    
    plt.suptitle(f"{exp['title']} - Multi-Seed Predictive Validity", fontsize=14)
    plt.tight_layout()
    save_dual_plot("13_predictive_validity.png", "validity", exp['id'])
    plt.close()