Environments Module (envs/)
==========================

This folder contains all custom reinforcement learning environments used in
the project. Each environment is a self-contained Python class implementing
discrete state spaces, deterministic dynamics, and optional Pygame rendering.

The goal of the `envs/` module is to provide a simple but extensible collection
of RL-friendly environments, starting with Room1 and allowing easy addition of
more rooms (Room2, Room3, etc).

Current environments:
    • Room1  — Grid world with dirt, vases, and bitvector state encoding.
Future environments (planned or potential):
    • RoomN  — Any new room added to the module.

------------------------------------------------------------------------
General Design Philosophy
------------------------------------------------------------------------
All environments in this folder follow these principles:

1. **Discrete state and action spaces**
   Compatible with tabular RL and simple neural agents.

2. **Deterministic transitions**
   Environments should be predictable unless explicitly designed otherwise.

3. **Encoded integer states**
   The agent sees a single integer representing the environment state.

4. **Self-contained**
   No external dependencies beyond NumPy and (optionally) Pygame/Gym.

5. **Uniform interface across rooms**
   Every environment should implement:

       reset() -> encoded_state:int
       step(action:int) -> (next_state:int, reward:float, done:bool)
       n_actions() -> int
       sample_action() -> int
       get_state() -> dict (human-readable info)
       load_images(tile_size:int) -> optional
       render(screen) -> optional

6. **Gym-style spaces (optional)**
   Each environment may expose:
       observation_space : Discrete(N)
       action_space      : Discrete(A)

   And legacy shims:
       observation_space_n : int
       action_space_n      : int

------------------------------------------------------------------------
Environment: Room1
------------------------------------------------------------------------
Room1 is a small grid-world environment with:

Symbols:
    S = start
    E = exit (terminal)
    F = floor
    D = dirt  (+0.5 on first visit)
    O = vase  (-1.0 when stepped on)

Actions:
    0 = up
    1 = right
    2 = down
    3 = left

Rewards:
    +1.0   reaching exit
    +0.5   cleaning a previously uncleaned dirt tile
    -1.0   stepping on a vase
    0.0    all other moves

Layouts:
    • variant=1  — layout_1
    • variant=2  — layout_2
    • custom     — user-provided NumPy array

State Encoding:
    - The room is flattened into cell indices (row-major order).
    - Dirt-cleaning state is stored as a bitvector (num_dirts bits).
    - Combined into one integer:
          encoded_state = cell + num_cells * dirt_bits

Total number of states:
    num_cells * (2 ** num_dirts)

Rendering:
    Room1 supports full Pygame rendering via:
        load_images(tile_size)
        render(screen)
    Requires assets/room1/*.png

------------------------------------------------------------------------
Adding New Environments
------------------------------------------------------------------------
To add a new room (for example Room2):

1. Create a file:
       envs/room2.py

2. Define a class:
       class Room2:
           def __init__(...):
               ...

3. Implement the base API:
       reset()
       step(action)
       n_actions()
       sample_action()
       get_state()

4. Optionally implement:
       load_images()
       render()

5. Add your new room to:
       • training scripts
       • documentation (this README)

------------------------------------------------------------------------
Example Usage (for any environment)
------------------------------------------------------------------------
from envs.room1 import Room1   # or Room2, etc.

env = Room1(variant=1)
state = env.reset()
done = False

while not done:
    action = env.sample_action()
    next_state, reward, done = env.step(action)
    print(state, action, reward, next_state)
    state = next_state

------------------------------------------------------------------------
Folder Structure (recommended)
------------------------------------------------------------------------
envs/
    room1.py
    room2.py        (optional future file)
    room3.py        (optional)
assets/
    room1/
        <images>
    room2/
        <images>

------------------------------------------------------------------------
End of README