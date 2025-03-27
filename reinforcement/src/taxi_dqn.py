'''
Tabular Q-learning works great for small, discrete environments like Taxi-v3. But when:
 * State/action spaces grow large or become continuous,
 * Or you want to generalize learning to unseen states,
...you need a neural network to approximate the Q-function:
Q(s,a)≈NeuralNetwork(s)[a]
'''

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-hot encode the discrete state
def one_hot(state, state_size=500):
    vec = torch.zeros(state_size, dtype=torch.float32)
    vec[state] = 1.0
    return vec.to(device)

# Neural network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, state_size=500, action_size=6):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn(n_episodes=2500, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.001, alpha=0.001):
    env = gym.make("Taxi-v3", render_mode=None)
    model = QNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    mse_loss_fn = nn.MSELoss()

    epsilon = epsilon_start
    rewards_per_episode = []

    window_size = 100          # moving average window
    check_every = 100          # check every N episodes
    patience = 5               # how many checks to tolerate no improvement
    epsilon_thresh = 1e-2      # how small a delta is considered "no improvement"
    best_avg = -float("inf")
    no_improve_count = 0

    for i in range(n_episodes):
        state, _ = env.reset()
        episode_done = False
        episode_reward = 0

        while not episode_done:

            # Choose next action 
            state_tensor = one_hot(state)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated
            next_state_tensor = one_hot(next_state)

            # Q-learning target
            with torch.no_grad():
                next_q = model(next_state_tensor)
                max_next_q = torch.max(next_q).item()
                done_mask = 1 if not episode_done else 0
                target_q = reward + gamma * max_next_q * done_mask

            # Predicted Q
            q_values = model(state_tensor)
            pred_q = q_values[action]
            #print(f'{q_values.detach().numpy()} {action} → {pred_q}')

            # Loss and backprop
            loss = mse_loss_fn(pred_q, torch.tensor(target_q).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            episode_reward += reward

        # At the end of this episode, update epsilon-greedy action selection
        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        rewards_per_episode.append(episode_reward)

        if i >= window_size and i % check_every == 0:
            moving_avg = np.mean(rewards_per_episode[-window_size:])
            
            if moving_avg > best_avg + epsilon_thresh:
                best_avg = moving_avg
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} checks (Avg Reward: {moving_avg:.2f})")

            if no_improve_count >= patience:
                print(f"Converged after {i} episodes (Avg Reward: {moving_avg:.2f})")
                break

        # Show the average reward over the last 100 episodes
        if i % 100 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"Episode {i:>5}: Avg Reward (last 100): {avg:.2f}, Epsilon: {epsilon:.3f}")

    env.close()
    # Plot results
    plt.plot(np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid'))
    plt.title("DQN: Moving Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward (100 ep)")
    plt.grid()
    plt.show()

    torch.save(model.state_dict(), "dqn_taxi.pth")
    print("Model saved to dqn_taxi.pth")            


    return model


def load_dqn_model(filepath="dqn_taxi.pth"):
    model = QNetwork().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Loaded model from {filepath}")
    return model


def run_dqn(filepath="dqn_taxi.pth", render_mode=True):
    env = gym.make("Taxi-v3", render_mode="human" if render_mode else None)
    model = load_dqn_model(filepath)

    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        state_tensor = one_hot(state)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward}")


train_dqn()
run_dqn()