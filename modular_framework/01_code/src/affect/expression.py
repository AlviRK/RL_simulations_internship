import pygame
import math

class Expression:

    def __init__(self):
        self.confidence=0
        self.curiosity=0
        self.insight=0

    def update(self, con, cur, ins): # expressed affects, not internal affects
        self.confidence=con
        self.curiosity=cur
        self.insight=ins * 3 #scale it so its visible
         
    def render(self, screen, env):
        x = env.current_pos % env.num_cols
        y = env.current_pos // env.num_cols
        size = env.TILE_SIZE

        #Draw the insight builb
        bulb_size=size*self.insight
        bulb = pygame.transform.scale(env.image_assets[f"bulb"], (bulb_size, bulb_size))
        screen.blit(bulb, (x * size + size/2 - bulb_size/2, y * size + size/2 - bulb_size/2))

        #Do the insght delay
        pygame.time.delay(int(self.insight*100))

        #express confidence

        #express curiosity
        

