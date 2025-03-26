import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

RNG = np.random.default_rng()


def _run_simulation(n_episodes, is_slippery, training_mode, render_mode):

    render_mode = 'human' if render_mode else None
    env = gym.make('FrozenLake-v1', 
                   map_name="8x8", 
                   is_slippery=is_slippery, 
                   render_mode=render_mode)

    if training_mode:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as fin:
            q = pickle.load(fin)

    alpha = 0.9 # learning rate
    gamma = 0.9 # discount rate
    epsilon = 1
    epsilon_decay_rate = 0.0001

    rewards_per_episode = np.zeros(n_episodes)
    for i in range(n_episodes):
        state = env.reset()[0] # states: 0 (TL corner) to 63 (BR corner)
        terminated = False     # True when fall through ice or reached destination
        truncated = False      # True when actions > max_episode_steps

        while not terminated and not truncated:
            if training_mode and RNG.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, info = env.step(action)
            if training_mode:
                q[state,action] = q[state,action] + alpha * (
                    reward + gamma * np.max(q[new_state,:]) - q[state,action])
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon==0:
            alpha = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

        if i % 1000 == 0:
            print(f'Finished episode {i:>5}...')

    env.close()

    last100_rewards = np.zeros(n_episodes)
    for t in range(n_episodes):
        last100_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(last100_rewards)
    plt.savefig('frozen_lake8x8.png')

    if training_mode:
        with open("frozen_lake8x8.pkl","wb") as fout:
            pickle.dump(q, fout)

def train(n_episodes, is_slippery=True):
    return _run_simulation(n_episodes, is_slippery, True, False)

def run(is_slippery=True):
    return _run_simulation(1, False, True)

if __name__ == '__main__':
    train(15000)
    #run()