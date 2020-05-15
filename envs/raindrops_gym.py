import gym
from gym import error, spaces, utils
import pygame
import numpy as np
import subprocess, time, sys
from envs.raindrops_python.game import Game
from envs.raindrops_python.sprites.raindrop import Raindrop


class RaindropsGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, show_score = False, pixels = 50):
        self.show_score = show_score
        self.game = Game()
        self.pixels = pixels
        self.action_space = spaces.Discrete((self.pixels * 2) + 1)

    def step_frame(self, action):
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
                self.game.bucket.move(0, action)
            else:
                self.game.bucket.move(1, action)

            for drop in self.game.raindrops:
                drop.fall()

            # Raindrop events
            for drop in self.game.raindrops:
                if drop.rect.colliderect(self.game.bucket.rect):
                    if drop.is_bad:
                        self.game.score -= self.game.score % 10
                    else:
                        self.game.score += 1
                    self.game.raindrops.remove(drop)
                    del drop
                elif drop.rect.bottomleft[1] > self.game.height:
                    if not drop.is_bad:
                        self.game.score -= 1
                    self.game.raindrops.remove(drop)
                    del drop

            # Drawing
            self.game.screen.fill((0,0,0))
            self.game.bucket.draw(self.game.screen)
            for drop in self.game.raindrops:
                drop.draw(self.game.screen)

            # Manage score    
            self.game.display_score()
            self.game.process_score()

            pygame.display.update()

    def step(self, action):
        if action == 0: # NO OP
            self.game.moveBucket(0,0)
        elif action <= self.pixels:
            self.game.moveBucket(0, action * 10)
        else:
            self.game.moveBucket(1, action * 10)

        reward = self.game.reward()
        observation = self.game.screen()
        game_over = self.game.state()

        observation = np.frombuffer(observation, dtype=np.int8)
        observation.shape = (480, 800, 4)

        return observation, reward, game_over, {}

    def reset(self):
        self.game.reset()
        return self.game.screen()

    def render(self, mode='human'):
        img = self.game.screen()
        arr = np.frombuffer(img, dtype=np.int8)
        arr.shape = (480, 800, 4)
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(arr)
        return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None