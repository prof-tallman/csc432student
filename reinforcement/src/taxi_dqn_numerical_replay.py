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
MODEL_FILE = "dqn_taxi_numerical_replay.keras"


def encode_state(state, taxi_env):
    """ Decode a Taxi state into four numerical input features. """

    row, col, passenger, destination = taxi_env.decode(state)

    features = [
        row / 4,                    # Normalized taxi row
        col / 4,                    # Normalized taxi column
        *np.eye(5)[passenger],      # One-hot passenger location
        *np.eye(4)[destination]     # One-hot destination
    ]

    return np.array([features], dtype=np.float32)


def create_model(state_size, action_size, alpha=0.001):
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
              epsilon_decay=0.0005, alpha=0.001,
              filepath=MODEL_FILE):

    # Create the environment and the neural network model
    env = gym.make(ENV_NAME)
    taxi_env = cast(TaxiEnv, env.unwrapped)
    state, _ = env.reset()
    assert isinstance(env.action_space, Discrete)
    n_features = len(encode_state(state, taxi_env).ravel())
    n_actions = env.action_space.n
    model = create_model(n_features, n_actions, alpha)

    # Create a target network with identical initial weights
    target_model = create_model(n_features, n_actions, alpha)
    target_model.set_weights(model.get_weights())

    # Initialize training parameters
    epsilon = epsilon_start  # for epsilon-greedy action policy
    rewards_per_episode = [] # to track learning progress
    window_size = 100        # average total reward over last 'n' episodes
    check_every = 100        # how often I want to check for early stopping
    patience = 5             # number checks to stop early
    best_avg = -float("inf") # best average reward so far
    no_improve_count = 0     # number checks w/o improvement
    target_update_every = 200  # synchronize every 1000 environment steps

    # Action/reward memory so that consider multiple moves together
    memory = deque(maxlen=10000)
    batch_size = 32
    train_every = 4
    step_count = 0

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
            # Must convert state number to an encoded array
            state_input = encode_state(state, taxi_env)
            q_values = model(state_input, training=False).numpy().ravel()

            # Choose an action using epsilon-greedy
            # This function is for training only
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values)

            # Take the action, observe the result, and store the experience
            next_state, reward, terminated, truncated, info = env.step(action)
            memory.append((state, action, reward, next_state, terminated))
            step_count += 1

            # Train every few actions, once we have enough experiences
            if len(memory) >= batch_size and step_count % train_every == 0:

                # Select a random batch of past experiences (creates tuples)
                # We select randomly, rather than recently, to avoid a problem
                #   'catastrophic forgetting' where we train too much on, say,
                #   the last few moves at the end of the episode instead of
                #   a selection of moves from early on where we potentially
                #   learn some bad actions to avoid
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, terminals = zip(*batch)

                # Convert states into encoded arrays
                state_inputs = np.vstack([encode_state(s, taxi_env) for s in states])
                next_inputs = np.vstack([encode_state(s, taxi_env) for s in next_states])

                # Run through NN to predict Q-values for all states in batch
                q_values = model(state_inputs, training=False).numpy()

                # Predict future Q-values using the target network
                next_q = target_model(next_inputs, training=False).numpy()

                # Calculate Q-learning targets (target is new desired Q value)
                # So it's not a NN in terms of supervised learning where the
                #   output values never change and we keep changing weights to
                #   create a system that generates expected values... instead
                #   it is a moving target as we figure out the expected future
                #   discounted reward for different actions from various states
                # The (1 - terminated) bit is because some actions result in
                #   terminating the episode so these just need to multiply the
                #   expected future gain by 0 and just take the current reward
                # Therefore, we turn terminated=True/False into terminated=1/0
                #   and if it did terminate, this term becomes a 0... if the
                #   episode continues it becomes a 1.
                rewards = np.array(rewards)
                discounted_future = gamma * np.max(next_q, axis=1)
                terminators = (1 - np.array(terminals, dtype=np.float32))
                targets = (rewards + discounted_future * terminators)

                # Update the target Q-value for each action in this batch
                q_values[np.arange(batch_size), actions] = targets

                # Train the network on the entire batch of historic actions
                model.train_on_batch(state_inputs, q_values)

            # Periodically copy the main network's weights to the target network
            if step_count % target_update_every == 0:
                target_model.set_weights(model.get_weights())

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
        if (i + 1) >= window_size and (i + 1) % check_every == 0:
            moving_avg = np.mean(rewards_per_episode[-window_size:])

            if moving_avg > best_avg + 0.01:
                best_avg = moving_avg
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Stopped after {i+1} episodes")
                break

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
    taxi_env = cast(TaxiEnv, env.unwrapped)
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
        taxi_row, taxi_col, passenger, destination = taxi_env.decode(state)
        locations = ["Red", "Green", "Yellow", "Blue"]

        # Run the episode
        while not terminated and not truncated:

            # The model returns a 2D tensor and we want a 1D array
            q_values = model(encode_state(state, taxi_env), training=False).numpy().ravel()

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
        print(f"    -> Taxi started at: ({taxi_row}, {taxi_col})")
        print(f"    -> Passenger at:    {locations[passenger]}")
        print(f"    -> Destination at:  {locations[destination]}")
        print(f"    -> Steps taken: {steps}")
        print(f"    -> Total reward: {total_reward}")

    # Finish the episodes
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