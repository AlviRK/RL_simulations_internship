import numpy as np
import pygame
from pathlib import Path
import random



class Room1:
    # Defining different room layouts
    TILE_SIZE=64

    LAYOUTS = {
    "1": np.array([
        list("SFFFFFFF"),
        list("FFFFFFFF"),
        list("FDFOFFFF"),
        list("FFFFFOFF"),
        list("FFFOFFDF"),
        list("FOOFFFOF"),
        list("FOFFOFOF"),
        list("FFFOFFFE"),
    ]),

    "2": np.array([
        list("SFFFFFFF"),
        list("FFFFFFFF"),
        list("FFFOFFFF"),
        list("FFFFFOFF"),
        list("FFFOFFDF"),
        list("FOOFFFOF"),
        list("FOFFOFOF"),
        list("FFFOFFFE"),
    ]),

    "5-state": np.array([
            list("SFFFE")
        ]),
        "7-state-mid-reward": np.array([
            list("SFFOFFE")
        ]),
        "10-state": np.array([
            list("SFFFFFFFFE")
        ])
    }

    # --- Initialization ---
    def __init__(self, * , variant="1", layout=None):
        # Choose a layout
        if layout is None:
            layout = self.LAYOUTS[variant]

        # Set chosen layout
        self.set_layout(layout)

        # Initialize the action space
        self.actions = (0, 1, 2, 3)  # 0=up, 1=right, 2=down, 3=left
        self.action_names = {0:"up", 1:"right", 2:"down", 3:"left"}
        self.num_actions = len(self.actions)
        
    def set_layout(self, layout):
        self.layout = np.array(layout) # set the right layout
        self.num_rows, self.num_cols = self.layout.shape # extract col and row info
        self.num_cells = self.num_rows * self.num_cols # extract cell number info


        # Parse layout and reset to initialize it for the first time
        self.parse_layout()
        self.agent_orientation = 1 # facing right intially
        self.current_pos = self.start_cell
        self.cleaned_dirts = []
        self.broken_vases = []




    # --- Parsing layout ---

    # Parse the layout to identify start, terminal, dirt and vase cells
    def parse_layout(self):
        layout = self.layout

        start, dirts, vases, terminal = None, [], [], None # initialize lists for dirt + vase positions, intiizalize start and terminal to None

        # Loop through layout array
        for i in range(layout.shape[0]):
            for j in range(layout.shape[1]):
                cell = str(layout[i, j]).upper() # Make cell a variable forcing uppercase and string (in case custom maps are poorly designed)
                idx = i * self.layout.shape[1] + j # Convert 2D index into 1D, row major
                if cell == "S":
                    start = idx
                elif cell == "E":
                    terminal = idx
                elif cell == "D":
                    dirts.append(idx)
                elif cell == "O":
                    vases.append(idx)
                elif cell == "F":
                    continue

        self.start_cell = start
        self.dirt_cells = dirts
        self.vase_cells = vases
        self.terminal_cell = terminal
        self.num_dirts = len(dirts)
        self.num_vases = len(vases)
        self.state_space_size = self.num_cells * (2 ** self.num_dirts) 
        self.observation_space_n = self.state_space_size # extract observation space size

    # --- Create Bit Representation of dirts ---

    # Convert tuples to integer representation
    def bits_to_int(self, b_tuple):
        bits = 0
        for k, bk in enumerate(b_tuple):
            if bk == 1:
                bits += 2**k
        return bits

    # Convert integer representation to tuples
    def int_to_bits(self, bits):
        b_tuple = []
        for k in range(self.num_dirts):
            if bits % (2 ** (k + 1)) >= 2 ** k:
                b_tuple.append(1)
            else:
                b_tuple.append(0)
        return tuple(b_tuple) # dirt status per index (cleaned=1, not cleaned=0)

    # --- Create State representation (including dirt bits) ---

    # Encode state as single integer
    def encode_state(self, cell, bits):
        encoded_state = cell + self.num_cells * bits
        return encoded_state
    
    # Decode state from single integer
    def decode_state(self, encoded_state):
        cell = encoded_state % self.num_cells
        bits = encoded_state // self.num_cells
        decoded_state = (cell, bits)
        return decoded_state
    
    # Get state function with all the info for later easy use
    def get_state(self):
        cell = self.current_pos
        b_tuple = tuple(1 if i in self.cleaned_dirts else 0 for i in self.dirt_cells)
        bits = self.bits_to_int(b_tuple)
        state = self.encode_state(cell, bits)
        state_tuple = (cell, *b_tuple)
        row, col = divmod(cell, self.num_cols)
        return {
            "state": state, # index, fully flattened
            "state_tuple": state_tuple, # (cell, dirt1, dirt2, ...)
            "cell": cell, # index 0 to num_cells-1
            "row": row, # row index
            "col": col, # column index
            "bits": bits, # dirt bitvector as integer
            "b_tuple": b_tuple # dirt bitvector as tuple
        }
        

    # --- Environment Dynamics ---

    # Step function
    def step (self, action):
        self.agent_orientation = action
        state_info = self.get_state()
        state   = state_info["state"]
        bits    = state_info["bits"]
        b_tuple = state_info["b_tuple"]
        state_tuple = state_info["state_tuple"]
        cell    = state_info["cell"]
        row     = state_info["row"]
        col     = state_info["col"]
    
        # Determine next position including boundary checks
        if action == 0: # up
            if row > 0:
                new_pos = cell - self.num_cols
            else:
                new_pos = cell
        elif action == 1: # right
            if col < self.num_cols - 1:
                new_pos = cell + 1
            else:
                new_pos = cell
        elif action == 2: # down
            if row < self.num_rows - 1:
                new_pos = cell + self.num_cols
            else:
                new_pos = cell
        elif action == 3: # left
            if col > 0:
                new_pos = cell - 1
            else:
                new_pos = cell

        self.current_pos = new_pos

        # Determine reward and done conditions
        if self.current_pos == self.terminal_cell:
            reward = 1.0
            done = True
        elif self.current_pos in self.vase_cells: # at the moment: Every time (vases are not flags)
            reward = -1.0
            done = False
            if self.current_pos not in self.broken_vases:
                self.broken_vases.append(self.current_pos)
        elif self.current_pos in self.dirt_cells and self.current_pos not in self.cleaned_dirts:
            reward = 0.5
            self.cleaned_dirts.append(self.current_pos)
            done = False
        else:
            reward = 0.0
            done = False

        next_state = self.get_state()["state"]

        return next_state, reward, done
    
    # Reset function
    def reset(self):
        self.current_pos = self.start_cell
        self.cleaned_dirts = []
        self.broken_vases = []
        self.agent_orientation = 1 # facing right initially
        state = self.get_state()["state"]
        return state

    def load_images(self, assets_subdir="room1"):
        self.tile_size = self.TILE_SIZE
        assets_dir = Path(__file__).resolve().parents[1] / "assets" / assets_subdir
        if not assets_dir.exists():
            raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

        def load(name):
            path = assets_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing asset: {path}")
            img = pygame.image.load(str(path))
            try:
                img = img.convert_alpha()
            except pygame.error:
                img = img.convert()
            return pygame.transform.scale(img, (self.tile_size, self.tile_size))
        
        self.image_assets = {
            "F": load("floor.png"),
            "D": load("dirt.png"),
            "O": load("obstacle.png"),
            "B": load("broken_obstacle.png"),
            "E": load("exit.png"),
            "S": load("start.png"),
            "agent_0": load("agent_up.png"),
            "agent_1": load("agent_right.png"),
            "agent_2": load("agent_down.png"),
            "agent_3": load("agent_left.png"),
            "bulb": load("bulb.png"),
        }

    def render(self, screen):
        
        for cell in range(self.num_cells):
            
            if cell == self.terminal_cell:
                img = self.image_assets["E"]
            elif cell == self.start_cell:
                img = self.image_assets["S"]
            elif cell in self.broken_vases:
                img = self.image_assets["B"]
            elif cell in self.vase_cells:
                img = self.image_assets["O"]
            elif cell in self.dirt_cells and cell not in self.cleaned_dirts:
                img = self.image_assets["D"]
            else:
                img = self.image_assets["F"]

            screen.blit(img, ((cell % self.num_cols) * self.tile_size, (1+ (cell // self.num_cols)) * self.tile_size))

        # Draw agent with orientation
        agent_img = self.image_assets[f"agent_{self.agent_orientation}"]
        screen.blit(agent_img, (self.current_pos % self.num_cols * self.tile_size, (1+(self.current_pos // self.num_cols)) * self.tile_size))
        

    def n_actions(self): 
        return self.num_actions

    def action_meanings(self): # to call for debugging
        return [self.action_names[a] for a in self.actions]

    def sample_action(self): # for testing; sample a random action
        return random.choice(self.actions)





