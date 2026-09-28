# Reinforcement learning framework — demo branch

For installation, execution and the meaning of the red/green signals, follow the [demo guide at the repository root](../README.md).

The entry point is **`01_code/src/run.py`**. The demo requires Python 3.13 and the packages in **`requirements.txt`**. It uses `affect/insight_ema.py` for the insight bulb and `affect/expression.py` for policy-change animations.

## Structure

```text
modular_framework/
├── requirements.txt           # Dependencies for the live demo
├── requirements-analysis.txt  # Optional analysis dependencies
├── 01_code/src/
│   ├── run.py                 # Main training and live animation script
│   ├── agents/                # TD and SARSA agents
│   ├── affect/                # Signals and expression
│   ├── envs/                  # Room1 environment
│   ├── assets/room1/          # Included robot and room images
│   └── replay/                # Replay support; logs created after training
└── 02_analysis/simulation/     # Existing analysis scripts
```

## Optional analysis

The analysis scripts are retained for reference and require additional packages. From this `modular_framework` directory, install them with:

```bash
python -m pip install -r requirements-analysis.txt
```

Before running `02_analysis/simulation/simulation_analysis/plot.py`, check its experiment configuration and the log/output paths in `utils.py`. The analysis workflow is separate from the verified live demo and may expect logs from additional experiments.

The old `run_conflict_exploration.py`, `run_conflict_relaxation.py`, `affect/insight.py` and `affect/insight_policy.py` are omitted from this branch. Their earlier versions remain in the development branch and Git history. `uncertainty.py` is retained because `run.py` computes and records it.
