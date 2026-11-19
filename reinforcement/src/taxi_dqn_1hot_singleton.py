import keras
import random
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from keras import layers
from gymnasium.spaces import Discrete
from gymnasium.envs.toy_text.taxi import TaxiEnv
from collections import deque
from typing import cast


ENV_NAME = "Taxi-v4"
MODEL_FILE = "dqn_taxi_1hot_singleton.keras"


def one_hot(state, state_size=500):
    """ Encodes a discrete state number into a one-hot vector. """

    # np.eye() generates an n-by-n identity matrix and we use slicing to select
    # a single row, with the 1 position corresponding to the state.
    #              [[ 1 0 0 0 0 ]
    #               [ 0 1 0 0 0 ]
    #   e.g.,       [ 0 0 1 0 0 ]
    # this one -->  [ 0 0 0 1 0 ]
    #               [ 0 0 0 0 1 ]]
    return np.eye(state_size, dtype=np.float32)[state:state+1]


def create_model(state_size, action_size, alpha=0.0001):
    """ Creates a neural network model to approximate a Q-table. """

    model = keras.Sequential([
        keras.Input(shape=(state_size,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=alpha),
        loss="mse"
    )
    return model


def train_dqn(n_episodes=2500, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.1,
              epsilon_decay=0.001, alpha=0.001,
              filepath=MODEL_FILE):

    # Create the environment and the neural network model
    env = gym.make(ENV_NAME)
    assert isinstance(env.observation_space, Discrete)
    assert isinstance(env.action_space, Discrete)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    model = create_model(n_states, n_actions, alpha)

    # Initialize training parameters
    epsilon = epsilon_start  # for epsilon-greedy action policy
    rewards_per_episode = [] # to track learning progress
    window_size = 100        # average total reward over last 'n' episodes
    check_every = 100        # how often I want to check for early stopping
    patience = 5             # number checks to stop early
    best_avg = -float("inf") # best loss so far
    no_improve_count = 0     # number episodes w/o improvement

    # Train over this many episodes (...unless we stop early)
    for i in range(n_episodes):

        # Reset the environment for this episode
        state, info = env.reset()
        truncated = False
        terminated = False
        episode_reward = 0

        # Run this episode
        while not terminated and not truncated:

            # Predict Q-values for the current state
            # Must convert state number to a one-hot encoded array
            state_input = one_hot(state)
            q_values = model(state_input, training=False).numpy().ravel()

            # Choose an action using epsilon-greedy
            # This function is for training only
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values)

            # Take the action, observe the result, and store the experience
            next_state, reward, terminated, truncated, info = env.step(action)

            # Q(s,a) <- r + gamma * max Q(s',a')
            # Different Q-equation because learning is now in the NN model
            # Unlike tabular Q-learning, the NN handles the actual update
            if terminated:
                target_q = reward
            else:
                next_input = one_hot(next_state)
                next_q = model(next_input, training=False).numpy().ravel()
                target_q = reward + gamma * np.max(next_q)

            # Train the neural network using the updated Q-values
            # Leaves the target values for the other actions unchanged
            q_values[action] = target_q
            model.train_on_batch(state_input, q_values.reshape(1, -1))

            # Update the state and accumulate the episode reward
            state = next_state
            episode_reward += float(reward)

        # Reduce exploration after each episode
        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        rewards_per_episode.append(episode_reward)

        # Report progress every 'n' episodes
        if (i + 1) % check_every == 0:
            avg = np.mean(rewards_per_episode[-window_size:])
            print(f"Episode {i+1:>5}: "
                  f"Avg Reward: {avg:.2f}, Epsilon: {epsilon:.3f}")

        # Monitor average reward for early stopping
        #if (i + 1) >= window_size and (i + 1) % check_every == 0:
        #    moving_avg = np.mean(rewards_per_episode[-window_size:])

        #    if moving_avg > best_avg + 0.01:
        #        best_avg = moving_avg
        #        no_improve_count = 0
        #    else:
        #        no_improve_count += 1

        #    if no_improve_count >= patience:
        #        print(f"Stopped after {i+1} episodes")
        #        break

    # Training is complete, close the environment
    env.close()

    # Plot the moving average of episode rewards
    if len(rewards_per_episode) >= window_size:
        moving_avg = np.convolve(
            rewards_per_episode,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        plt.plot(np.arange(window_size, len(rewards_per_episode) + 1),
                 moving_avg)

    plt.title("Deep Q-Learning: Taxi-v4")
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Reward ({window_size} episodes)")
    plt.grid()
    plt.show()

    model.save(filepath)
    print(f"Model saved to {filepath}")

    return model


def run_dqn(n_episodes=5, filepath=MODEL_FILE, render_mode=True):

    # Load the trained model and tell Pylance to treat it as a keras.Model
    # Assumes a trained model was saved to a file
    env = gym.make(ENV_NAME, render_mode="human" if render_mode else None)
    model = keras.models.load_model(filepath)
    model = cast(keras.Model, model)

    for n in range(n_episodes):

        # Reset the environment and episode variables
        state, info = env.reset()
        steps = 0
        total_reward = 0
        terminated = False
        truncated = False

        # Keep track of the conditions for this rollout
        taxi_env = cast(TaxiEnv, env.unwrapped)
        taxi_row, taxi_col, passenger, destination = taxi_env.decode(state)
        locations = ["Red", "Green", "Yellow", "Blue"]

        # Run the episode
        while not terminated and not truncated:

            # The model returns a 2D tensor and we want a 1D array
            q_values = model(one_hot(state), training=False).numpy().ravel()

            # Select the best action and take a step
            action = np.argmax(q_values)
            state, reward, terminated, truncated, _ = env.step(action)
            if action == 5 and reward == -1 and not terminated:
                print("Passenger dropped off at the wrong destination!")        
            total_reward += float(reward)
            steps += 1

        # Report the results
        if terminated:
            print(f"{n+1}. Agent completed the task successfully!")
        else:
            print(f"{n+1}. Agent exhausted steps and timed out")
        print(f"  -> Taxi started at: ({taxi_row}, {taxi_col})")
        print(f"  -> Passenger at:    {locations[passenger]}")
        print(f"  -> Destination at:  {locations[destination]}")
        print(f"  -> Steps taken: {steps}")
        print(f"  -> Total reward: {total_reward}")

    # Finish the episode
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Deep Q-Learning for Taxi-v4 using Keras"
    )

    parser.add_argument("--train", action="store_true",
                        help="Train a new model instead of running rollout")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training or rollout episodes")
    parser.add_argument("--file", type=str, default=MODEL_FILE,
                        help="File for saving or loading the model")

    args = parser.parse_args()

    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be a positive integer")

    if args.train:
        n_episodes = args.episodes if args.episodes is not None else 2500
        train_dqn(n_episodes=n_episodes, filepath=args.file)
    else:
        n_episodes = args.episodes if args.episodes is not None else 5
        run_dqn(n_episodes=n_episodes, filepath=args.file)