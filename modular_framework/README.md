# Reinforcement Learning & Affective Computing Framework

This framework is designed for the simulation, analysis, and visualization of Reinforcement Learning (RL) agents (TD/SARSA) in gridworld environments (`Room1`), combined with affective signal modeling (Insight, Confidence, Curiosity, Expression).

---

## 📁 Project Structure & Architecture

The project is split into two main sections: **Simulation/Training** and **Data Analysis**.

```text
AffSim/
├── 01_code/                               # Simulation & Training directory
│   └── src/
│       ├── agents/                        # RL Agents (td_agent.py, sarsa_agent.py)
│       ├── envs/                          # Environments (room1.py)
│       ├── affect/                        # Affective models (insight.py, confidence.py, etc.)
│       ├── replay/
│       │   └── logs/                      # Data logs (seed_*.json) are saved HERE
│       └── train.py                       # Main script for training & live rendering
│
├── 02_analysis/                           # Analysis & Plotting directory
│   └── simulation/
│       ├── output/
│       │   └── demos/                     # Generated plots are saved HERE (by date)
│       └── simulation_analysis/
│           ├── plot.py                    # Main plotter (loads JSONs, triggers sub-analyses)
│           ├── utils.py                   # Path management (Single Source of Truth) & save logic
│           ├── baseline_analysis.py       # Learning curves
│           ├── dynamics_analysis.py       # Micro-traces & heatmaps
│           ├── event_analysis.py          # Discovery metrics
│           └── validity_analysis.py       # Validity checks


🛠️ Installation & Setup
Using a virtual Conda environment is highly recommended to ensure clean execution.

Create and activate the environment:


conda create -n affsim python=3.10
conda activate affsim

Install dependencies:
Use the provided requirements.txt file:
pip install -r requirements.txt


🚀 Workflows (How to use the framework)
1. Training & Live Rendering (01_code/src/train.py)

This script trains the agents across different random seeds and logs all behavioral data as well as affective signals into .json files.

Enable Live Rendering: Set render=True inside the main() block. A Pygame window will open upon execution.

Sample Configuration for Testing:


SEEDS = [42] # Use a single seed to visually inspect the behavior
train_and_save(..., render=True, fps=15, signals=True)
Controls During Live Rendering: While the Pygame window is active, press the Up Arrow Key on your keyboard to increase the simulation speed (FPS) or the Down Arrow Key to decrease it.

Production Run (for paper data): Set render=False and enable all scientific seeds (SEEDS = [42, 43, 44, 45, 46]) to generate statistically valid data logs in 01_code/src/replay/logs/.


2. Data Analysis & Plotting (02_analysis/.../plot.py)

Once the simulation data has been generated, this script performs statistical evaluations on the JSON files and generates the figures for the paper.

Execution:


cd 02_analysis/simulation/simulation_analysis
python plot.py
Output: Plots are automatically sorted and exported into 02_analysis/simulation/output/demos/[Date]/ (e.g., 190526).

Path Convention: All paths are dynamically resolved relative to the script location via utils.py. Manually update the date string there whenever you want to start a new experimental folder.