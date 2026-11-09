
# Thanks to Johnny Code for helping me understand the Q(s,a) equation and
# providing some simple demos with gymnasium. Johnny helped me to practically
# understand some theoretical concepts that I had read about elsewhere. Mostly
# that Q(s,a) means that we access Table Q at column s and action a *instead*
# being a series of values Q at some time t (s0,a0) -> (s1,a1) -> ... (st,at).
# https://www.youtube.com/watch?v=9fAnzZ6xzhA

# Official documentation:
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse

# Random seed used to make training reproducible
rngseed = 432

def _run_simulation(n_episodes:int, 
                    is_slippery:bool, 
                    training_mode:bool, 
                    file_name:str):
    '''
    Simple Q-Learning demonstration using Gymnasium's "Frozen Lake" game. This
    function is the core reinforcement learning algorithm that can be used for
    training and rollout. There should be two wrapper functions `train` and
    `rollout` that provide an easier interface to this function.

    Parameters:
     * n_episodes:  Number of iterations to train the model. Higher numbers
                    provide more learning but they take longer.
     * is_slippery: Adds a random element to the game. The agent has an 33% of
                    moving in the intended direction but there's a 66% chance
                    that it 'slips' and goes a perpindicular direction.
     * training_mode: Determines whether the model should train by creating
                    and saving a new Q-Table or make use of an existing model.
     * file_name:   Saves the trained model under this file name (relative to
                    the working directory).

    Returns: The Q-Table as a numpy array
    '''

    RNG = np.random.default_rng(seed=rngseed)

    # Converts render_mode using terms that gymnasium understands
    render_mode = None if training_mode else 'human'

    # Create the gym environment
    env = gym.make('FrozenLake8x8-v1', 
                   is_slippery=is_slippery, 
                   render_mode=render_mode)
    env.action_space.seed(rngseed)

    # Training mode starts with an empty Q-Table whereas rollout mode uses
    # a pretrained model that was saved to disk
    if training_mode:
        assert isinstance(env.observation_space, Discrete)
        assert isinstance(env.action_space, Discrete)
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(file_name, 'rb') as fin:
            q_table = pickle.load(fin)

    # Hyperparameters for the main Q-Learning Equation Q(s,a) and also for the 
    # epsilon-greedy policy that starts with 100% exploration and slowly moves
    # to 100% exploitation (i.e., experience) but only if there are enough
    # episodes to reach epsilon==0 at the given decay rate.
    alpha = 0.1        # learning rate
    gamma = 0.99       # discount rate
    epsilon = 1        # start at 100% exploration
    epsilon_min = 0.05 # never completely stop learning
    epsilon_decay_rate = 0.00001

    # Rewards are overly simplistic for this 8x8 environment. If elf reaches
    # the final square in the bottom-right corner (square #63), then the agent
    # receives a reward of 1. Every other state/action receives a score of 0.
    rewards_history = np.zeros(n_episodes)

    # Iterate through each episode, one a time.
    for i in range(n_episodes):
        state, info = env.reset(seed=rngseed if training_mode and i == 0 else None)
        terminated = False
        truncated = False

        # Run a single episode until it ends.
        while not terminated and not truncated:

            # In training mode, we use epislon greedy to choose between a 
            # random action and a learned action.
            if training_mode:
                if RNG.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state,:])

            # In rollout mode, we always exploit the learning by selecting
            # the action with the highest expected future reward.
            else:
                action = np.argmax(q_table[state,:])

            # Perform the action in the environment and receive back the new
            # state information, the reward, and whether or not the episode is
            # complete. If we're in training mode, we update the Q-Table.
            next_state, reward, terminated, truncated, info = env.step(action)
            if training_mode:
                future_value = 0
                if not terminated:
                    future_value = gamma * np.max(q_table[next_state,:])
                learned = alpha * (float(reward) + future_value - q_table[state,action])
                q_table[state, action] += learned
            state = next_state

        # Report the outcome of each rollout episode
        if not training_mode:
            if reward == 1:
                print(f"Episode {i+1}: SUCCESS (reached the goal)")
            elif truncated:
                print(f"Episode {i+1}: TRUNCATED (step limit reached)")
            else:
                print(f"Episode {i+1}: FAILURE (fell into a hole)")

        # Update the epsilon-greedy algorithm so that we gradually change from
        # exploration to exploitation (learning).
        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

        # Save the last reward value and only the last reward value. It will
        # have a value of 1 if we ended on square #63. Or, it will have a
        # value of 0 if we ended on any other square. The episode ended either
        # because we reached square #63, fell through the lake, or timed out.
        rewards_history[i] = reward

        # Helpful debug output
        if training_mode and (i + 1) % 10000 == 0:
            print(f'Finished training episode {i+1:>6}...')

    # Release all resources for this environment
    env.close()

    if not training_mode:
        wins = int(np.sum(rewards_history))
        success_rate = wins / n_episodes
        print(f"Rollout: {wins}/{n_episodes} wins ({success_rate:.1%})")

    # Plot the reward results for the last 100 episodes. Since the reward is
    # either 0 or 1 for any given episode, the sum of the last 100 episodes
    # will roughly correspond to the win rate for the agent.
    if training_mode:
        last100_rewards = (
            pd.Series(rewards_history).
            rolling(window=100, min_periods=1).
            mean()
        )
        plt.figure()
        plt.plot(last100_rewards)
        plt.ylabel("Success Rate (last 100 episodes)")
        plt.xlabel("Episode")
        plt.savefig(f'{file_name}.png')
        plt.close()

    # Save the Q-Table to 
    if training_mode:
        actions = ["←", "↓", "→", "↑"]
        df = pd.DataFrame(q_table, columns=actions)
        df.round(3).to_csv(f'{file_name}.csv')

    # Save the fully trained model.
    if training_mode:
        with open(file_name,"wb") as fout:
            pickle.dump(q_table, fout)

    return q_table


def train(n_episodes:int = 10000, 
          is_slippery:bool = True, 
          file_name:str = "frozen_lake8x8.pkl"):
    '''
    Trains a Q-Learning model in Gymnasium's Frozen Lake environment. The
    final model is saved to disk with an image showing the learning rate.

    Parameters:
     * n_episodes:  Number of iterations to train the model. Higher numbers
                    provide more learning but they take longer.
     * is_slippery: Adds a random element to the game. The agent has an 33% of
                    moving in the intended direction but there's a 66% chance
                    that it 'slips' and goes a perpindicular direction.
     * file_name:   Saves the trained model under this file name (relative to
                    the working directory).

    Returns: The Q-Table as a numpy array
    '''
    return _run_simulation(n_episodes, is_slippery, True, file_name)


def rollout(n_episodes:int = 5,
            is_slippery:bool = True, 
            file_name:str = "frozen_lake8x8.pkl"):
    '''
    Visualizes one episode in the environment with a fully trained Q-Learning
    model.

    Parameters
     * is_slippery: Adds a random element to the game. The agent has an 33% of
                    moving in the intended direction but there's a 66% chance
                    that it 'slips' and goes a perpindicular direction.
     * file_name:   Loads the trained model from a Pickle file with this file
                    name (relative to the working directory).

    Returns: The Q-Table as a numpy array
    '''
    return _run_simulation(n_episodes, is_slippery, False, file_name)


def visualize_policy(q_table:np.ndarray, map_shape:tuple=(8, 8)):
    """
    Visualizes the learned policy as arrows in a grid. Each arrow shows the 
    best action (←, ↓, →, ↑) for the agent in that state.

    Parameters:
     * q_table:   The learned Q-table (a NumPy array)
     * map_shape: Shape of the Frozen Lake map (default 8x8)
    """
    action_symbols = ['←', '↓', '→', '↑']
    policy_grid = np.full(map_shape, ' ', dtype=str)
    for state in range(q_table.shape[0]):
        row = state // map_shape[1]
        col = state % map_shape[1]
        if np.sum(q_table[state]) > 0:
            best_action = np.argmax(q_table[state])
            policy_grid[row, col] = action_symbols[best_action]
        else:
            policy_grid[row, col] = ' '

    print("\nLearned Policy:")
    for row in policy_grid:
        print(' '.join(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or run a Q-learning agent on Frozen Lake (8x8).'
    )
    parser.add_argument('--train', action='store_true',
                        help='Train and save a new agent (default: run saved agent).')
    parser.add_argument('--slippery', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Enable or disable slipping (default: slippery).')
    parser.add_argument('--file', default='frozen_lake8x8.pkl',
                        help='Q-table filename to save/load.')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes (default: 100000 training, 5 rollout).')
    args = parser.parse_args()

    if args.episodes is not None and args.episodes <= 0:
        parser.error('--episodes must be a positive integer')

    if args.train:
        episodes = args.episodes if args.episodes is not None else 100000
        q_table = train(episodes, args.slippery, args.file)
        visualize_policy(q_table)
    else:
        with open(args.file, 'rb') as fin:
            q_table = pickle.load(fin)
        visualize_policy(q_table)
        print()
        episodes = args.episodes if args.episodes is not None else 5
        rollout(episodes, args.slippery, args.file)
