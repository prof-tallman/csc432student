
# https://gymnasium.farama.org/environments/toy_text/taxi/

from itertools import product
import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# global random number generator
RNG = np.random.default_rng()

# Action Space (6 possibilities)
#  0: south (move down)
#  1: north (move up)
#  2: east  (move right)
#  3: west  (move left)
#  4: pickup passenger
#  5: drop off passenger

# Observation Space (25x5x4 => 500 total states)
#  25 taxi positions: a 5x5 grid with some walls
#  5 passenger locations
#     0: red
#     1: green
#     2: yellow
#     3: blue
#     4: in taxi
#  4 destinations
#     0: red
#     1: green
#     2: yellow
#     3: blue
#
# From the docs: An observation is returned as an int() that encodes the
#   corresponding state, calculated by:
#     = ((taxi_row * 5 + taxi_col) * 5 
#       + passenger_location) * 4
#       + destination

# Rewards
#  -1 per step unless other reward is triggered.
#  +20 delivering passenger.
#  -10 executing “pickup” and “drop-off” actions illegally.

def _run_simulation(n_episodes:int, 
                    training_mode:bool, 
                    file_name:str):
    '''
    Simple Q-Learning demonstration using Gymnasium's "Taxi" game. This is the
    core reinforcement learning algorithm that can be used for training and
    rollout. There should be two wrapper functions `train` and `rollout` that
    provide an easier interface to this function.

    Parameters:
     * n_episodes:  Number of iterations to train the model. Higher numbers
                    provide more learning but they take longer.
     * training_mode: Determines whether the model should train by creating
                    and saving a new Q-Table or make use of an existing model.
     * render_mode: Selects whether to show and animate the episodse or run
                    them all in the background. If `training_mode` is True,
                    this parameter is usually False (it's **much** faster).
                    But if `training_mode` is False, the user should decide
                    whether they want to visually see the results or just use
                    the text calculations.
     * file_name:   Saves the trained model under this file name (relative to
                    the working directory).

    Returns: The final Q-Table in a Numpy array
    '''

    # Converts render_mode True/False to terms that gymnasium understands
    render_mode = None if training_mode else 'human'
    env = gym.make('Taxi-v3', render_mode=render_mode)

    # Training mode starts with an empty Q-Table whereas rollout mode uses
    # a pretrained model that was saved to disk
    if training_mode:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(file_name, 'rb') as fin:
            q_table = pickle.load(fin)

    # Hyperparameters for the main Q-Learning Equation Q(s,a) and also for the 
    # epsilon-greedy policy that starts with 100% exploration and slowly moves
    # to 100% exploitation (i.e., experience) but only if there are enough
    # episodes to reach epsilon==0 at the given decay rate.
    alpha = 0.9 # learning rate (very high)
    gamma = 0.9 # discount rate (mid-range)
    epsilon = 1 # start at 100% exploration
    epsilon_decay_rate = 0.0001

    rewards_history = np.zeros(n_episodes)    
    for i in range(n_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        rewards = 0
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
                tempdif = gamma * np.max(q_table[next_state,:])
                learned = alpha * (reward + tempdif - q_table[state,action])
                q_table[state,action] = q_table[state,action] + learned
            state = next_state
            rewards += reward

        # Update the epsilon-greedy algorithm so that we gradually change from
        # exploration to exploitation (learning).
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # After finishing training, lower the learning rate for stability.
        if epsilon==0:
            alpha = 0.0001

        rewards_history[i] = reward

        # Helpful debug output
        if training_mode and i % 1000 == 0:
            if i == 0:
                print(f'Finished training episode {i+1:>5}...')
            else:
                print(f'Finished training episode {i:>5}...')

    # Release all resources for this environment.
    env.close()

    # Plot the moving average over 100 episodes.
    if training_mode:
        window_size = 100
        kernel_100 = np.ones(window_size)/window_size
        moving_avg = np.convolve(rewards_history, kernel_100, mode='valid')
        plt.figure()
        plt.plot(moving_avg)
        plt.title("Moving Average of Cumulative Rewards (Window=100)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{file_name}.png')
        plt.close()

    # Save the fully trained model.
    if training_mode:
        with open(file_name,"wb") as fout:
            pickle.dump(q_table, fout)

    # Save the Q-table as a CSV
    if training_mode:
        actions = [ 'South', 'North', 'East', 'West', 'Pickup', 'Dropoff' ]
        states = _generate_observation_list()
        df = pd.DataFrame(q_table, index=states, columns=actions)
        df.round(3).to_csv(f"{file_name}.csv")

    return q_table


def _generate_observation_list():
    '''
    Creates a descriptive observation list to use as the index for a Q-Table.
    '''
    # Based on observation Space (25x5x4 => 500 total states)
    #  25 taxi positions: a 5x5 grid with some walls
    #  5 passenger locations: 0-red, 1-green, 2-yellow, 3-blue, 4-in taxi
    #  4 destinations:        0-red, 1-green, 2-yellow, 3-blue
    plist = 'RGYBT'
    dlist = 'RGYB'
    observations = []
    for taxi_row, taxi_col, passenger_location, destination in (
            product(range(5), range(5), range(5), range(4))):
        value = (( taxi_row * 5 + taxi_col ) * 5 
                    + passenger_location) * 4 + destination
        ptxt = plist[passenger_location]
        dtxt = dlist[destination]
        help_str = f'({taxi_col}, {taxi_row}) {ptxt}->{dtxt}'
        observations.append((value, help_str))
    observations.sort(key=lambda x: x[0])
    observations = [s for idx, s in observations]
    return observations


def train(n_episodes:int = 2000, 
          file_name:str = "taxi.pkl"):
    '''
    Trains a Q-Learning model in Gymnasium's Taxi environment. The final model
    is saved to disk along with the Q-Table and an image of the learning rate.

    Parameters:
     * n_episodes:  Number of iterations to train the model. Higher numbers
                    provide more learning but they take longer.
     * file_name:   Saves the trained model under this file name (relative to
                    the working directory).

    Returns: The final Q-Table in a Numpy array
    '''
    return _run_simulation(n_episodes, True, file_name)


def rollout(file_name:str = "taxi.pkl"):
    '''
    Visualizes one episode in the Taxi environment with a fully trained
    Q-Learning model.

    Parameters
     * file_name:   Loads the trained model from a Pickle file with this file
                    name (relative to the working directory).

    Returns: The final Q-Table in a Numpy array
    '''
    return _run_simulation(1, False, file_name)


if __name__ == '__main__':
    train(3000)
    input("Press <ENTER> to rollout the model")
    rollout()




















    #interactive_rollout()

"""
def interactive_rollout(file_name: str = "taxi.pkl"):
    '''
    Allows a human to manually play the Taxi-v3 environment using keyboard input.
    Shows the environment and lets the user pick actions at each step.

    Parameters:
     * file_name: Loads the trained model from a Pickle file with this name.

    Returns: None
    '''
    env = gym.make("Taxi-v3", render_mode="human")

    # Load the trained model (optional, not used for decisions here)
    with open(file_name, 'rb') as fin:
        q_table = pickle.load(fin)

    state, info = env.reset()
    terminated = False
    truncated = False

    actions = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    print("\n--- Interactive Taxi-v3 ---")
    print("Controls: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff")
    print("Or type 'best' to use the model's best action for the current state.")
    print("Type 'exit' to quit.\n")

    while not terminated and not truncated:
        try:
            # Ask for action
            user_input = input("Your move (0-5, best, exit): ").strip().lower()
            if user_input == 'exit':
                break
            elif user_input == 'best':
                print(np.round(q_table[state, :], 3))
                action = int(np.argmax(q_table[state, :]))
                print(f"Model chooses: {action} ({actions[action]})")
            else:
                action = int(user_input)
                if action < 0 or action >= 6:
                    print("Invalid action. Try again.")
                    continue
                print(f"You chose: {action} ({actions[action]})")

            # Take the action
            next_state, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward}")
            state = next_state

        except ValueError:
            print("Invalid input. Please enter a number 0-5, 'best', or 'exit'.")

    env.close()
    print("Game over.")
"""