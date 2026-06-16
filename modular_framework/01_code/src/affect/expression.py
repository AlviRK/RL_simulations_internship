import pygame
import math

class Expression:

    def __init__(self):
        self.confidence = 0
        self.curiosity = 0
        self.insight = 0
        self.conflict = 0
        self.policy_changed = 0
        self.uncertainty = 0

    def update(self, con, cur, ins, conflict=0, policy_changed=0, uncertainty=0):
        self.confidence = con
        self.curiosity = cur
        self.insight = ins * 3
        self.conflict = conflict
        self.policy_changed = policy_changed
        self.uncertainty = uncertainty

    def render(self, screen, env):
        x = env.current_pos % env.num_cols
        y = env.current_pos // env.num_cols
        size = env.TILE_SIZE

        bulb_size = int(size * self.insight)

        if bulb_size > 0:
            bulb = env.image_assets["bulb"].copy()

            if self.policy_changed == 1:
                bulb.fill((0, 255, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            elif self.conflict == 1:
                bulb.fill((255, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            elif self.uncertainty ==1:
                bulb.fill((255, 160, 80, 255), special_flags=pygame.BLEND_RGBA_MULT) #(170, 100, 50, 255) == "Brown color"
          
            

            bulb = pygame.transform.scale(bulb, (bulb_size, bulb_size))
            screen.blit(
                bulb,
                (x * size + size / 2 - bulb_size / 2,
                 y * size + size / 2 - bulb_size / 2)
            )

        pygame.time.delay(int(self.insight * 100))