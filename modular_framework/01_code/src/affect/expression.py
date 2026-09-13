import numpy as np
import pygame


class Expression:
    RED_MOVE_MS = 420
    GREEN_PREVIEW_MS = 360
    GREEN_FADE_MS = 120
    REAL_MOVE_MS = 300
    ARRIVAL_HOLD_MS = 70

    def __init__(self):
        self.confidence = 0
        self.curiosity = 0
        self.insight = 0
        self.policy_changed = 0
        self.new_greedy_action = None

    def update(self, con, cur, ins, ins_pos=0, ins_neg=0, policy_changed=0, new_greedy_action=None): #stores the latest signal values
        self.confidence = con
        self.curiosity = cur
        self.insight = ins * 3
        self.policy_changed = policy_changed
        self.new_greedy_action = new_greedy_action

    def render(self, screen, env, *, pause=True, position=None, opacity=255): #draws the bulb above the robot.
        size = env.TILE_SIZE
        if position is None:
            row, col = divmod(env.current_pos, env.num_cols) #returns the row and column where the agent is
            position = ((col + 0.5) * size, (row + 1.5) * size) #center of the tail

        bulb_size = int(size * self.insight)
        if bulb_size > 0:
            bulb = pygame.transform.scale(
                env.image_assets["bulb"], (bulb_size, bulb_size)
            )
            bulb.set_alpha(opacity)
            screen.blit(bulb, (
                position[0] - bulb_size / 2,
                position[1] - size - bulb_size / 2,
            ))
        if pause: #it temporarily pauses the agent depending on the insight value
            pygame.time.delay(int(self.insight * 100))

    @staticmethod
    def policy_signal(q_before, q_after, action): #decides whether to show green, red, or no animation
        
        best_before = q_before == np.max(q_before) #It compares each value of q_before with the max value and gives True/False
        best_after = q_after == np.max(q_after)
        
        # Green: there is one preferred action afterward, and the preferred set changed.
        if np.count_nonzero(best_after) == 1 and not np.array_equal(best_before, best_after):
            return int(np.flatnonzero(best_after)[0]), (50, 210, 80), False 
           
        # Red: the action lost value and is no longer preferred.
        # There are several potential replacements tied. A unique replacement would already have returned green above!
        if best_before[action] and not best_after[action] and q_after[action] < q_before[action]: 
            return int(action), (230, 60, 60), True
        return None

    def animate_policy_signal(
        self, screen, env, state, action, color, excluded=False,
        *, executed_action=None, before_background=None,
    ): #runs the animation chosen by the previous function.

        size = env.TILE_SIZE
        origin = env.decode_state(state)[0]
        real_cell = env.current_pos
        if executed_action is None:
            executed_action = env.agent_orientation

        def center(cell):
            return pygame.Vector2(
                (cell % env.num_cols + 0.5) * size,
                (cell // env.num_cols + 1.5) * size,
            )

        canvas = pygame.Surface(screen.get_size(), depth=24)
        clock = pygame.time.Clock()
        a_pos, b_pos = center(origin), center(real_cell)

        # Extract the robot's shape so we can color it without coloring its white background.
        def silhouette(orientation):
            source = env.image_assets[f"agent_{orientation}"]
            rgb = pygame.surfarray.array3d(source)
            alpha = pygame.surfarray.array_alpha(source).astype(np.uint16)
            return (255 - rgb.min(axis=2)).astype(np.uint16) * alpha // 255

        masks = {orientation: silhouette(orientation)
                 for orientation in {executed_action, action}}

        # Draw one animation frame: the background, the robot at the given position, direction, color and opacity, and its bulb. 
        def frame(position, orientation, tint, opacity=255):
            pygame.event.pump()
            if pygame.event.peek(pygame.QUIT):
                return False
            canvas.blit(before_background, (0, 0))
            mask = masks[orientation]
            rect = pygame.Rect(0, 0, *mask.shape)
            rect.center = (round(position.x), round(position.y))
            clipped = rect.clip(canvas.get_rect())
            if clipped.width and clipped.height:
                mx, my = clipped.x - rect.x, clipped.y - rect.y
                coverage = mask[mx:mx + clipped.width, my:my + clipped.height]
                coverage = (coverage * opacity // 255)[..., None]
                pixels = pygame.surfarray.pixels3d(canvas)
                region = pixels[clipped.x:clipped.right, clipped.y:clipped.bottom]
                try:
                    old = region.astype(np.uint16)
                    ink = np.asarray(tint, dtype=np.uint16)
                    region[:] = (old * (255 - coverage) + ink * coverage + 127) // 255
                finally:
                    del region
                    del pixels
            self.render(canvas, env, pause=False, position=position, opacity=opacity)
            screen.blit(canvas, (0, 0))
            pygame.display.flip()
            return True
     
        # Repeated calls with different positions create the movement. Move the robot smoothly from start to end over the given duration, optionally fading it out.
        def phase(start, end, orientation, tint, duration, fade=False):
            elapsed = 0
            while elapsed < duration:
                t = elapsed / duration
                eased = t * t * (3 - 2 * t)
                opacity = round(255 * (1 - eased)) if fade else 255
                if not frame(start.lerp(end, eased), orientation, tint, opacity):
                    return False
                elapsed += max(1, clock.tick(60))
            return frame(end, orientation, tint, 0 if fade else 255)

        # If the red condition is met, move the robot from A to B in red,
        # then keep it briefly at B in red before returning to black.
        if excluded:
            if not phase(a_pos, b_pos, executed_action, color, self.RED_MOVE_MS):
                return False
            if not phase(b_pos, b_pos, executed_action, color, self.ARRIVAL_HOLD_MS):
                return False
      
        # Otherwise, find C in the newly preferred direction, keeping it within the board,
        # and animate the robot from A to C in green.
        else:
            row, col = divmod(origin, env.num_cols)
            dc, dr = ((0, -1), (1, 0), (0, 1), (-1, 0))[action]
            next_col = max(0, min(env.num_cols - 1, col + dc))
            next_row = max(0, min(env.num_rows - 1, row + dr))
            c_pos = center(next_row * env.num_cols + next_col)
            if not phase(a_pos, c_pos, action, color, self.GREEN_PREVIEW_MS):
                return False
           # If the newly preferred action is the same action the robot took… Then the green movement already represents the real move, so it doesn’t need to repeat it in black.
            if action == executed_action:
                if not phase(b_pos, b_pos, executed_action, color, self.ARRIVAL_HOLD_MS):
                    return False
           
        # This runs when the newly preferred action differs from the action actually taken.         
            else:
                if not phase(c_pos, c_pos, action, color, self.GREEN_FADE_MS, fade=True):
                    return False #the green robot stays at C and fades out. It already moved there earlier.
                if not phase(a_pos, b_pos, executed_action, (0, 0, 0), self.REAL_MOVE_MS): #the robot moves from A to B in black, showing the action actually taken.
                    return False

        canvas.fill((0, 0, 0))
        env.render(canvas)
        self.render(canvas, env, pause=False)
        screen.blit(canvas, (0, 0))
        pygame.display.flip()
        elapsed = 0
        while elapsed < self.ARRIVAL_HOLD_MS:
            pygame.event.pump()
            if pygame.event.peek(pygame.QUIT):
                return False
            elapsed += max(1, clock.tick(60))
        return True

    def animate_policy_ghost(self, screen, env, state, action, color, excluded=False, **kwargs):
        return self.animate_policy_signal(screen, env, state, action, color, excluded, **kwargs)
