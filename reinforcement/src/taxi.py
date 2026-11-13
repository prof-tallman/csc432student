# Official documentation:
# https://gymnasium.farama.org/environments/toy_text/taxi/

# Action Space (6 possibilities)
#  0: south (move down)
#  1: north (move up)
#  2: east  (move right)
#  3: west  (move left)
#  4: pickup passenger
#  5: drop off passenger

# Observation Space (25 x 5 x 4 = 500 total states)
#  25 taxi positions: a 5x5 grid with some walls
#  5 passenger locations: 0=red, 1=green, 2=yellow, 3=blue, 4=in taxi
#  4 destinations:        0=red, 1=green, 2=yellow, 3=blue
#  state = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

# Rewards
#  -1  per step unless another reward is triggered
#  +20 for delivering the passenger
#  -10 for illegal pickup or drop-off

from itertools import product
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
                    training_mode:bool,
                    file_name:str):
    '''
    Simple Q-Learning demonstration using Gymnasium's "Taxi" game. This
    function is the core reinforcement learning algorithm that can be used for
    training and rollout. The wrapper functions `train` and `rollout` provide
    an easier interface to this function.

    Parameters:
     * n_episodes:   Number of episodes to train or run.
     * training_mode: Train a new Q-Table or use a saved model.
     * file_name:    File name used to save or load the trained Q-Table.

    Returns: The Q-Table as a NumPy array
    '''

    RNG = np.random.default_rng(seed=rngseed)

    # Converts render_mode using terms that gymnasium understands
    render_mode = None if training_mode else 'human'

    # Create the Gymnasium environment.
    env = gym.make('Taxi-v4', render_mode=render_mode)
    env.action_space.seed(rngseed)

    # Training starts with an empty Q-Table. Rollout loads a saved Q-Table.
    if training_mode:
        assert isinstance(env.observation_space, Discrete)
        assert isinstance(env.action_space, Discrete)
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(file_name, 'rb') as fin:
            q_table = pickle.load(fin)

    # Hyperparameters for Q-learning and epsilon-greedy action selection.
    # We start with exploration and gradually favor learned actions.
    alpha = 0.9          # learning rate (high - learn quickly)
    gamma = 0.9          # discount factor (value future rewards)
    epsilon = 1          # start at 100% exploration
    epsilon_min = 0.05   # continue to explore occasionally during training
    epsilon_decay_rate = 0.0001   # faster decay rate

    # Taxi accumulates rewards at every step. The total reward per episode
    # shows whether the taxi is becoming faster and avoiding bad actions.
    rewards_history = np.zeros(n_episodes)
    successes = 0

    # Iterate through episodes, one at a time.
    for i in range(n_episodes):
        state, info = env.reset(seed=rngseed if i == 0 else None)
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0

        # Run a single episode until it terminates or reaches the step limit.
        while not terminated and not truncated:

            # During training, epsilon-greedy chooses between exploration
            # (random action) and exploitation (best learned action).
            if training_mode:
                if RNG.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state,:])

            # During rollout, always choose the best learned action.
            else:
                action = np.argmax(q_table[state,:])

            # Take an action, collect the reward, and update the Q-Table.
            next_state, reward, terminated, truncated, info = env.step(action)
            if training_mode:
                future_value = 0
                if not terminated:
                    future_value = gamma * np.max(q_table[next_state, :])
                learned = alpha * (float(reward) + future_value - q_table[state, action])
                q_table[state, action] += learned

            state = next_state
            episode_reward += float(reward)
            steps += 1

        # Taxi terminates successfully when the passenger is delivered.
        if terminated:
            successes += 1

        if not training_mode:
            outcome = 'SUCCESS (passenger delivered)' if terminated else 'TRUNCATED (step limit reached)'
            print(f'Episode {i+1}: {outcome}; {steps} steps; total reward {episode_reward:g}')

        # Gradually move from exploration toward exploitation during training.
        if training_mode:
            epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

        # Save the cumulative reward, rather than just the final-step reward.
        rewards_history[i] = episode_reward

        # Helpful debug output during longer training runs.
        if training_mode and (i + 1) % 10000 == 0:
            print(f'Finished training episode {i+1:>6}...')

    env.close()

    if not training_mode:
        print(f'Rollout: {successes}/{n_episodes} successful deliveries '
              f'({successes/n_episodes:.1%}); '
              f'average total reward {np.mean(rewards_history):.1f}')

    # Plot the mean cumulative reward over the last 100 episodes.
    if training_mode:
        last100_rewards = (
            pd.Series(rewards_history).
            rolling(window=100, min_periods=1).
            mean()
        )
        plt.figure()
        plt.plot(last100_rewards)
        plt.ylabel('Average Total Reward (last 100 episodes)')
        plt.xlabel('Episode')
        plt.savefig(f'{file_name}.png')
        plt.close()

    # Save the Q-Table with descriptive state names for classroom inspection.
    if training_mode:
        actions = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
        states = _generate_observation_list()
        df = pd.DataFrame(q_table, index=states, columns=actions)
        df.round(3).to_csv(f'{file_name}.csv')

    # Save the trained model.
    if training_mode:
        with open(file_name, 'wb') as fout:
            pickle.dump(q_table, fout)

    return q_table


def _generate_observation_list():
    '''Creates descriptive labels for all 500 states of the Taxi Q-Table.'''
    plist = 'RGYBT'   # Passenger location: Red, Green, Yellow, Blue, Taxi
    dlist = 'RGYB'    # Destination: Red, Green, Yellow, Blue
    observations = []

    for taxi_row, taxi_col, passenger_location, destination in (
            product(range(5), range(5), range(5), range(4))):
        value = ((taxi_row * 5 + taxi_col) * 5
                 + passenger_location) * 4 + destination
        ptxt = plist[passenger_location]
        dtxt = dlist[destination]
        help_str = f'({taxi_col}, {taxi_row}) {ptxt}->{dtxt}'
        observations.append((value, help_str))

    observations.sort(key=lambda x: x[0])
    return [label for idx, label in observations]


def train(n_episodes:int = 30000,
          file_name:str = 'taxi.pkl'):
    '''
    Train a Q-learning model in Taxi and save its Q-Table, CSV, and reward plot.
    Returns the Q-Table as a NumPy array.
    '''
    return _run_simulation(n_episodes, True, file_name)


def rollout(n_episodes:int = 5,
            file_name:str = 'taxi.pkl'):
    '''
    Visualize episodes using a previously trained Taxi Q-Table.
    Returns the Q-Table as a NumPy array.
    '''
    return _run_simulation(n_episodes, False, file_name)


def visualize_policy(q_table:np.ndarray,
                     passenger_location:int = 0,
                     destination:int = 1):
    '''
    Show one 5x5 slice of the learned policy for a fixed passenger location
    and destination. Taxi has 500 states, so a single grid cannot show every
    possible passenger/destination combination.

    The default shows passenger at Red (R), destination Green (G).
    Action symbols: ↓ ↑ → ← P (pickup) D (drop-off).
    '''
    action_symbols = ['↓', '↑', '→', '←', 'P', 'D']
    location_names = ['Red', 'Green', 'Yellow', 'Blue', 'In taxi']
    policy_grid = np.full((5, 5), ' ', dtype='<U1')

    for taxi_row in range(5):
        for taxi_col in range(5):
            state = ((taxi_row * 5 + taxi_col) * 5
                     + passenger_location) * 4 + destination
            if np.any(q_table[state] != 0):
                policy_grid[taxi_row, taxi_col] = action_symbols[np.argmax(q_table[state])]

    print(f'\nLearned Policy (passenger: {location_names[passenger_location]}, '
          f'destination: {location_names[destination]}):')
    print('P=Pickup, D=Drop-off; blank=unlearned state')
    for row in policy_grid:
        print(' '.join(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or run a Q-learning agent on Taxi (5x5).'
    )
    parser.add_argument('--train', action='store_true',
                        help='Train and save a new agent (default: run saved agent).')
    parser.add_argument('--file', default='taxi.pkl',
                        help='Q-table filename to save/load.')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes (default: 30000 training, 5 rollout).')
    args = parser.parse_args()

    if args.episodes is not None and args.episodes <= 0:
        parser.error('--episodes must be a positive integer')

    if args.train:
        episodes = args.episodes if args.episodes is not None else 30000
        q_table = train(episodes, args.file)
        visualize_policy(q_table)
    else:
        with open(args.file, 'rb') as fin:
            q_table = pickle.load(fin)
        visualize_policy(q_table)
        print()
        episodes = args.episodes if args.episodes is not None else 5
        rollout(episodes, args.file)
