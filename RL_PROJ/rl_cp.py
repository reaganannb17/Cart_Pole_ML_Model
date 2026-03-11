# =====================
# CartPole Game
# =====================

# Imports
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# =====================
# Replay Buffer
# =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =====================
# Q-Network
# =====================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), #hidden layer
            nn.ReLU(),
            nn.Linear(128, 128), #hidden layer
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# =====================
# Epsilon-Greedy Policy
# =====================
def select_action(policy_net, state, epsilon, action_dim, device):
    if random.random() < epsilon:
        # Explore
        return random.randrange(action_dim)
    else:
        # Exploit
        state_v = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape (1, state_dim)
        with torch.no_grad():
            q_values = policy_net(state_v)
        return q_values.max(1)[1].item()


# =====================
# Training Step
# =====================
def compute_dqn_loss(batch, policy_net, target_net, gamma, device):
    states, actions, rewards, next_states, dones = batch

    states_v      = torch.FloatTensor(states).to(device)
    actions_v     = torch.LongTensor(actions).to(device)
    rewards_v     = torch.FloatTensor(rewards).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    dones_v       = torch.FloatTensor(dones).to(device)

    # Q(s, a) for the actions we actually took
    q_values = policy_net(states_v)  # [batch_size, action_dim]
    state_action_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Target: r + γ * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q_values = target_net(next_states_v)
        max_next_q_values = next_q_values.max(1)[0]
        target_values = rewards_v + gamma * max_next_q_values * (1.0 - dones_v)

    # MSE loss
    loss = nn.MSELoss()(state_action_values, target_values)
    return loss


# =====================
# Main Training Loop
# =====================
def train_dqn_cartpole(
    env_name="CartPole-v1",
    num_episodes=500,
    replay_size=10000,
    batch_size=64,
    gamma=0.99,
    learning_rate=1e-3,
    sync_target_steps=1000,
    epsilon_start=1.0,
    epsilon_final=0.01,
    epsilon_decay=500,
    render=False,
):
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_size)

    steps_done = 0
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0

        while True:
            if render:
                env.render()

            # Epsilon decays over time
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                      np.exp(-1.0 * steps_done / epsilon_decay)

            action = select_action(policy_net, state, epsilon, action_dim, device)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            # Update Q-network if enough samples
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_dqn_loss(
                    batch, policy_net, target_net, gamma, device
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Periodically sync target network
            if steps_done % sync_target_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        mean_reward = np.mean(episode_rewards[-20:])

        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Mean(20): {mean_reward:.1f} | Epsilon: {epsilon:.3f}"
        )

        # Early stopping criterion: consistently good performance
        if mean_reward >= 475.0 and len(episode_rewards) >= 20:
            print("Environment solved! Stopping early.")
            break

    env.close()
    return policy_net, episode_rewards


if __name__ == "__main__":
    trained_net, rewards = train_dqn_cartpole(
        env_name="CartPole-v1",
        num_episodes=500,
        replay_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=1e-3,
        sync_target_steps=1000,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
        render=True,    
    )
