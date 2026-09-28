# Policy-change animation demo

This `demo` branch includes the red and green policy-change animations and the insight bulb. 

## Download

With Git installed, open a terminal (Anaconda Prompt on Windows) and run:

```bash
git clone --branch demo --single-branch https://github.com/AlviRK/RL_simulations_internship.git rl-policy-demo
cd rl-policy-demo
```

Alternatively, [download this branch as a ZIP](https://github.com/AlviRK/RL_simulations_internship/archive/refs/heads/demo.zip), extract it, and open a terminal in the extracted folder containing this README and `modular_framework`.

## Install and run

These commands use Conda (Anaconda or Miniconda). Run them from the repository folder:

```bash
conda create -n rl-policy-demo python=3.13 -y
conda activate rl-policy-demo
python -m pip install -r modular_framework/requirements.txt
python modular_framework/01_code/src/run.py
```

If you already use virtual environments without Conda, create a fresh environment with Python 3.13, activate it, and run the last two commands. No previous training logs or downloaded data are needed; the robot images are included.

A Pygame window opens with the robot, dirt, vases and exit. Keep the window focused to use the up/down arrow keys to increase/decrease the normal step speed. Close the window to stop.

The default run uses seed 42, room variant 1, 300 episodes, at most 100 steps per episode, and 5 normal steps per second. Animations have their own timing. A full run takes a while at this speed. To try fewer episodes, change `episodes=300` in `main()` at the bottom of `run.py`.

Logs are written to `modular_framework/01_code/src/replay/logs/seed_42.json` when training finishes. Closing the window early ends the program without saving that partial run. Running the same seed again replaces its log.

## What the animations mean

- **Green:** after learning, there is one preferred action and the set of preferred actions has changed. If this is the action just executed, its movement is shown once in green. If it is a different action, the robot previews that direction in green, fades out, then shows the actual move in black from the original cell.
- **Red:** the action just executed was previously among the preferred actions and is no longer preferred. Several alternatives remain tied, so there is no unique replacement to show in green. The actual move is shown in red.
- **Black:** normal movement. Receiving a penalty does not necessarily trigger red: an action already outside the preferred set can be selected again through exploration.
- **Bulb:** its size reflects the smoothed magnitude of the TD error, indicating how much the learning target differs from the current estimate.

Here, "preferred" means having the highest learned Q-value in that state. Colors describe a change in preference, rather than simply positive or negative reward. The preview is only visual: it does not execute an extra environment step or change learning.

## Files to start with

- `modular_framework/01_code/src/run.py`: training loop and animation trigger.
- `modular_framework/01_code/src/affect/expression.py`: red/green conditions and animation.
- `modular_framework/01_code/src/affect/insight_ema.py`: insight signal for the bulb.
- `modular_framework/01_code/src/envs/room1.py`: room, rewards and robot images.

`uncertainty.py`, confidence, curiosity, SARSA and replay support are retained because the current program uses or imports them. The `02_analysis` folder and existing notebooks are background material, not required to run this demo. See the [framework notes](modular_framework/README.md) for optional analysis dependencies.

## Verification

Checked on macOS with Python 3.13.13, NumPy 2.4.4 and Pygame 2.6.1 in an isolated environment. The checks cover 300 training episodes, unchanged training results, all three animation paths with the included images, and window-close handling. Graphics were tested using an offscreen SDL display; this does not constitute a Windows or Linux desktop test.

Earlier experiments remain available in `feature/modular-framework` and the Git history. This demo does not replace that development branch.
