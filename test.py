import gym, time, random
import numpy as np
from envs.raindrops_gym import *

env = RaindropsGym('/Users/noah/Documents/Raindrop/desktop/build/libs/desktop-1.0.jar', 5)
env = RaindropsWrapper(
    env, 
    n_bins=8, 
    low=[-2.4, -2.0, -0.42, -3.5], 
    high=[2.4, 2.0, 0.42, 3.5]
)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

print('q-table made')

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(0, 100000):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward < 0:
            penalties += 1

        state = next_state
        epochs += 1
        print("step")

    print(f"Episode: {i}")
        
    if i % 100 == 0:
        np.save('q_table.npy', q_table)

print("Training finished.\n")