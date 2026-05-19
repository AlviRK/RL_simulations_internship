# Data Logging Architecture

The training process captures data at three levels: Environment/Agent metadata, Training session metadata, and Episode-specific results.

### 1. Master Data Container (`data`)
To ensure reproducibility and context, all results are stored within a single master dictionary. This acts as the "Source of Truth" for the entire training run.
- **env_meta**: Logs the world layout (rows, cols, dirt/vase placement).
- **agent**: Records the agent's identity and hyper-parameters (alpha, gamma, epsilon).
- **train_meta**: Captures session constraints (total episodes, max steps, fps).
- **replays**: A list containing all recorded episode entries.

### 2. Episode Recording (`replay_entry`)
Data generated during a single episode is captured in a `replay_entry` dictionary. This dictionary is re-initialized every episode to ensure a clean slate.
- **Structure**:
    - **Step-aligned arrays**: `actions`, `states`, `next_states`, `rewards`, `td_error`, `policy_changes`, `insight`, `confidence`, `curiosity`.
    - **Events**: `events` (a list of specific milestones reached during steps).
    - **State Visits**: `state_visits` (a dictionary mapping states to their frequency of visitation).
    - **Summary**: A nested dictionary for high-level analysis.

### 3. Summary Nesting
The `summary` dict provides a "quick-look" at the episode's success without needing to iterate through the bulky step-aligned arrays.
- **Keys**: `first_dirt_step`, `second_dirt_step`, `first_vase_step`, `exit_step`, `revisit_reward_steps`, `num_dirts_found`, `num_vases_hit`, `steps`, `total_reward`.

### 4. Storage Logic
- Each `replay_entry` is appended directly to `data["replays"]`.
- The entire `data` dictionary is serialized to a JSON file and returned by the training function, ensuring the agent's performance is always bundled with its configuration.

### 5. File Structure & Serialization 
All data from a training session is serialized into a single JSON file. 
- **Filename**: `training_data.json` (formerly `replays.json`) 
- **Format**: A single root dictionary containing metadata and a results list.


### 6. Experiment Configuration 
Before visualization, the analysis script utilizes a "Control Center" configuration (`EXPERIMENTS`) to map raw data to visual labels. 
- **Experiment Registry**: Defines the `id`, `title`, and specific `signal` (e.g., "Insight v01") to be analyzed. 
- **Path Resolution**: The system automatically checks for data in both hierarchical (`/logs/id/`) and flat directory structures to ensure backward compatibility. 

### 7. Data Loading & Seed Attribution 
The loader performs two transformations: 
- **Seed Injection**: The loader extracts the `seed` from the `agent` metadata and injects it into every individual `replay_entry`.
- **Aggregation**: The loader standardizes the input into a list of dictionaries, allowing the analysis modules to iterate through multiple seeds/files seamlessly. 
- **Plot Styling**: All outputs are standardized using `seaborn-v0_8-paper` and `colorblind`palettes to ensure publication-quality results.

---
# Data Analysis
## Baseline Analysis: Task Performance
The baseline analysis evaluates the agent's fundamental learning progress and efficiency through two primary metrics.

### 1. Learning Curve (Return Plot)
#### Purpose
This plot tracks the cumulative reward per episode to demonstrate that the agent is learning to solve the task. A rising curve indicates that the agent is successfully optimizing its policy to maximize rewards over time.

#### Calculation & Data Flow
- **Logging**: During training, rewards are summed per episode into `total_reward`.
- **Storage**: This value is stored in the `summary` nested dictionary within each `replay_entry` , which is appended to the `data`dictionary under the key "replays".
- **Aggregation**: The analysis script extracts the `total_reward` from the `replays` list across all seeds and calculates the mean and 95% confidence interval (CI) for each episode.

### 2. Efficiency Curve (Steps Plot)
#### Purpose
This plot tracks the number of steps taken to reach the goal or terminate the episode. It serves as a measure of efficiency; as the agent learns the optimal path, the number of steps should generally decrease and stabilize.

#### Calculation & Data Flow
- **Logging**: The `steps` counter tracks every action taken by the agent until the episode ends or `max_steps` is reached.
- **Storage**: The final step count is stored in the `summary` nested dictionary of the `replay_entry`.
- **Aggregation**: The analysis script compiles these counts to visualize how quickly the agent "solves" the environment across different training episodes.

### Analysis Logic
The baseline analysis:
- **Seed Handling**: Iterates through multiple training seeds to account for stochasticity in exploration.
- **Data Transformation**: Converts the JSON log structure into a long-form DataFrame (Episode, Return/Steps, Seed) suitable for relational plotting.
- **Visualization**: Uses line plots with shaded error bands representing the 95% confidence interval.

---
## Descriptive Analysis


#### Purpose
The descriptive analysis serves to understanding the structure and patterns of the signal: 
What does the signal look like normally? Where and when does it occur? How does it look like over time?
moves from "How much did the agent learn?" to "How did the agent experience the episode?" 
By visualizing specific episodes (Beginner, Intermediate, and Expert), we can observe the relationship between Insight and learning progress over time.

### 3. Insight Dynamics (Micro-Analysis)

#### Purpose
Examining insight occurrences in specific episodes (start, middle, and end of training). It is designed to visualize the temporal relationship between "Insight" spikes, reward accumulation, and specific environment events (like hitting a vase or finding dirt) within each selected episode.

#### Calculation & Data Flow
- **Logging**: During training, insight values are logged per step into `insight_val`, which is appended per episode into `insight_buffer`.
- **Storage**: This value is stored within each `replay_entry`, which is appended to the `data`dictionary under the key "replays". 
- **Aggregation**: The plot.py script's main function pours all data across seeds into an `all_seeds` variable.
-
- **Selection**: Sample three representative snapshots from a single seed: the first episode, the median episode, and the final episode.


#### Analysis Logic
- **Seed handling**: One specific seed (the first) is selected because this analysis focuses on single episodes within one seed (Micro-Analysis) and poured into a `session_data`variable.
- **Data Transformation**: The data under the "replays" key in `session_data` are poured into the `replays`variable. 
-  **Temporal Sampling**: For micro-analysis, the script calculates indices to extract snapshots of the agent's training by dividing the total episode count `total_eps` (e.g., Episode 0, Episode $N/2$, and Episode $N-1$).
- **Signal Extraction**:
	- **Insight**: The script pulls the step-aligned data using `r.get('insight', [])` into a `signal` variable
	- **Rewards & Events**: Raw step-wise rewards are retrieved via `r.get('rewards', [])` into `rewards`, while environmental milestones are pulled from the `events` list into `events`. **Temporal Alignment**: A `steps` array is created using `np.arange(len(signal))`. This provides a "Pythonic" guarantee that the x-axis index perfectly aligns with the length of the captured signals, regardless of episode duration. 
	- **Cumulative Reward**: The script performs a `np.cumsum(rewards)` to transform individual step rewards into a `cum_reward` variable.
- **Dual-Axis Plotting**: Uses a twin-axis layout to plot the magnitude of the Insight signal (step-based, $y_1$) versus the growth of Reward (over the selected, $y_2$). Additionally, non-zero insight occurrences are drawn as blue dots.  
- **Event Alignment**: Synchronizes spatial milestones (Events) with the agent's internal state (Insight). This allows for visual verification of whether Insight spikes correlate with significant environmental interactions.
- **Naming**: Generates a stacked multi-plot (`03_insight_micro_stacked.png`) saved into a descriptive output folder based on the unique `experiment_id`.


### 4. Insight Trend (Convergence)

#### Purpose
Examining the development of Insight (per episode mean and max) over all episodes and contrast it against episode return. 