import numpy as np
import gym

from time import sleep
import random


if __name__ == '__main__':
    # hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    env = gym.make('Taxi-v2').env # environment
    q_table = np.zeros([env.observation_space.n, env.action_space.n]) # Q table

    n_episodes = 100000 # number of episodes

    for i in range(n_episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, done, info = env.step(action)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            
            if reward == -10:
                penalties += 1
            state = next_state
            epochs += 1
        if i % 100 == 0:
            print(f'Episode: {i}')
    print('Training finished.\n')
