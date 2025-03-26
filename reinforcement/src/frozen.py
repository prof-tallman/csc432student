
# Thanks to Johnny Code for helping me understand the Q(s,a) equation and
# providing some simple demos with gymnasium. Johnny helped me to practically
# understand some theoretical concepts that I had read about elsewhere. Mostly
# that Q(s,a) means that we access Table Q at column s and action a *instead*
# being a series of values Q at some time t (s0,a0) -> (s1,a1) -> ... (st,at).
# https://www.youtube.com/watch?v=9fAnzZ6xzhA

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# global random number generator
RNG = np.random.default_rng()


def _run_simulation(n_episodes:int, slippery:bool, 
                    training_mode:bool, render_mode:bool, filename:str):
    '''
    Simple Q-Learning demonstration using Gymnasium's "Frozen Lake" game. This
    function is the core reinforcement learning algorithm that can be used for
    training and rollout. There should be two wrapper functions `train` and
    `rollout` that provide an easier interface to this function.
    '''

    # Converts render_mode True/False to terms that gymnasium understands
    render_mode = 'human' if render_mode else None
    env = gym.make('FrozenLake-v1', 
                   map_name="8x8", 
                   is_slippery=slippery, 
                   render_mode=render_mode)

    # Training mode starts with an empty Q-Table whereas rollout mode uses
    # a pretrained model that was saved to disk
    if training_mode:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as fin:
            q = pickle.load(fin)

    # Hyperparameters for the main Q-Learning Equation Q(s,a) and also for the 
    # epsilon-greedy policy that starts with 100% exploration and slowly moves
    # to 100% exploitation (i.e., experience) but only if there are enough
    # episodes to reach epsilon==0 at the given decay rate.
    alpha = 0.9 # learning rate
    gamma = 0.9 # discount rate
    epsilon = 1 # start at 100% exploration
    epsilon_decay_rate = 0.0001

    # Rewards are overly simplistic for this 8x8 environment. If elf reaches
    # the final square in the bottom-left corner (square #63), then the agent
    # receives a reward of 1. Every other state/action receives a score of 0.
    rewards_history = np.zeros(n_episodes)
    
    for i in range(n_episodes):
        state, info = env.reset() # States: 0 (TL corner) to 63 (BR corner)
        terminated = False        # True when fall through ice or solved maze
        truncated = False         # True when actions > max_episode_steps

        while not terminated and not truncated:

            # In training mode, we use epislon greedy to choose between a 
            # random action and a learned action.
            if training_mode:
                if RNG.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state,:])

            # In rollout mode, we always exploit the learning by selecting
            # the action with the highest expected future reward.
            else:
                action = np.argmax(q[state,:])

            # Perform the action in the environment and receive back the new
            # state information, the reward, and whether or not the episode is
            # complete. If we're in training mode, we update the Q-Table.
            next_state, reward, terminated, truncated, info = env.step(action)
            if training_mode:
                q[state,action] = q[state,action] + alpha * (
                    reward + gamma * np.max(q[next_state,:]) - q[state,action])
            state = next_state

        # Update the epsilon-greedy algorithm so that we gradually change from
        # exploration to exploitation (learning).
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # After finishing training, lower the learning rate for stability
        if epsilon==0:
            alpha = 0.0001

        # Save the last reward value and only the last reward value. It will
        # have a value of 1 if we ended on square #63. Or, it will have a
        # value of 0 if we ended on any other square. The episode ended either
        # because we reached square #63, fell through the lake, or timed out.
        rewards_history[i] = reward

        # Helpful debug output
        if training_mode and i % 1000 == 0:
            if i == 0:
                print(f'Finished trianing episode {i+1:>5}...')
            else:
                print(f'Finished trianing episode {i:>5}...')

    # Release all resources for this environment
    env.close()

    # Plot the reward results for the last 100 episodes. Since the reward is
    # either 0 or 1 for any given episode, the sum of the last 100 episodes
    # will roughly correspond to the win rate for the agent.
    last100_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):
        last100_rewards[t] = np.sum(rewards_history[max(0, i-100):(i+1)])
    plt.plot(last100_rewards)
    plt.savefig('frozen_lake8x8.png')

    # Save the fully trained model.
    if training_mode:
        with open("frozen_lake8x8.pkl","wb") as fout:
            pickle.dump(q, fout)

def train(n_episodes, is_slippery=True, filename="frozen_lake8x8.pkl"):
    '''
    Trains a Q-Learning model in Gymnasium's Frozen Lake environment.

    Final model is saved to disk with an image showing the learning rate.
    '''
    return _run_simulation(n_episodes, is_slippery, True, False, filename)

def rollout(is_slippery=True, filename="frozen_lake8x8.pkl"):
    '''
    Visualizes a fully trained Q-Learning model.
    '''
    return _run_simulation(1, is_slippery, False, True, filename)

if __name__ == '__main__':
    train(15000)
    input("Press <ENTER> to rollout the model")
    rollout()