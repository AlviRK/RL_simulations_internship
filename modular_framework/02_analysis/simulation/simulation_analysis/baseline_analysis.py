import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import save_dual_plot





# ----------------------------------------------------------------
## A. Baseline Analysis: Task performance & explainability metrics
# ----------------------------------------------------------------

### Task Performance Sanity Panel: Learning Curves
def plot_baseline(all_seeds, exp):
    print("     [1/4] Analyzing Baseline")
    # 1. Convert everything into a long-form DataFrame for Seaborn
    rows = []
    for seed_data in all_seeds:
        seed_id = seed_data['replays'][0].get('seed', 'unknown')
        for r in seed_data['replays']:
            rows.append({
                "Episode": r['episode'],
                "Return": r.get('summary', {}).get('total_reward'),
                "Steps": r.get('summary', {}).get('steps', 0),
                "Seed": seed_id
            })
    df = pd.DataFrame(rows)

    # 1. Return Plot
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="Episode", y="Return", errorbar=('ci', 95))
    plt.title(f"{exp['title']} - Learning Curve (Mean + 95% CI)")
    save_dual_plot("01_learning_curve_return.png", "baseline", exp['id'])
    plt.close()

    # 2. Steps Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="Steps", color='orange', errorbar=('ci', 95))
    plt.title(f"{exp['title']} - Steps to Completion (Mean + 95% CI)")
    save_dual_plot("01_learning_curve_steps.png", "baseline", exp['id'])
    plt.close()