import gym
from gym import error, spaces, utils
import pygame
import numpy as np
import subprocess, time, sys
from envs.raindrops_python.game import Game
from envs.raindrops_python.sprites.raindrop import Raindrop

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

class RaindropsGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, show_score = False, pixels = 5):
        self.show_score = show_score
        self.game = Game()
        self.pixels = pixels
        self.action_space = spaces.Discrete((self.pixels * 2) + 1)

    def step(self, action, state):
        if self.game.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit() # quit the screen
                    sys.exit()
                    game.running = False

            # Manage in-game time
            self.game.loop_in_game_time += 1/60
            if self.game.loop_in_game_time > self.game.spawn_threshold:
                self.game.raindrops.append(Raindrop())
                self.game.loop_in_game_time = 0

            # Bucket moving
            if action == 0: # NO OP
                pass
            elif action <= self.pixels:
                self.game.bucket.move(0, (action % self.pixels) * 10)
            else:
                self.game.bucket.move(1, (action % self.pixels) * 10)

            for drop in self.game.raindrops:
                drop.fall()

            reward = .1
            # Raindrop events
            for drop in self.game.raindrops:
                if drop.rect.colliderect(self.game.bucket.rect):
                    if drop.is_bad:
                        self.game.score -= self.game.score % 10
                        reward = -1
                    else:
                        self.game.score += 1
                        reward = 1
                    self.game.raindrops.remove(drop)
                    del drop
                elif drop.rect.bottomleft[1] > self.game.height:
                    if not drop.is_bad:
                        self.game.score -= 1
                        reward = -1
                    self.game.raindrops.remove(drop)
                    del drop

            # Drawing
            self.game.screen.fill((0,0,0))
            self.game.bucket.draw(self.game.screen)
            for drop in self.game.raindrops:
                drop.draw(self.game.screen)

            # Manage score
            if self.show_score:  
                self.game.display_score()
            self.game.process_score()

            # Gym data
            image_data = self.observe(state)
            game_over = self.game.game_over
        return image_data, reward, game_over, {}

    def observe(self, state):
        x_t1_colored = pygame.surfarray.array3d(pygame.display.get_surface())
        x_t1 = rgb2gray(x_t1_colored)
        x_t1 = resize(x_t1,(100, 60))
        x_t1 = rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        return np.append(x_t1, state[:, :, :, :3], axis=3)

    def no_op(self):
        x_t_colored = pygame.surfarray.array3d(pygame.display.get_surface())
        x_t = rgb2gray(x_t_colored)
        x_t = resize(x_t,(100, 60))
        x_t = rescale_intensity(x_t, out_range=(0, 255))

        x_t = x_t / 255.0

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

        return s_t

    def reset(self):
        self.game.reset()
        return self.observe()

    def render(self, mode='human', close=False):
        pygame.display.update()

    def close(self):
        self.game.running = False