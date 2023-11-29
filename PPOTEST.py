import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

import gym



class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        print(f"POLICY_FORWARD: {state.shape}")
        return self.model(state)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_space_size, action_space_size, hidden_dim, learning_rate, gamma, epochs, clip_epsilon, batch_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epochs = epochs
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size

        self.policy_net = PolicyNetwork(state_space_size, hidden_dim, action_space_size)
        self.value_net = ValueNetwork(state_space_size, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def train(self):
        for _ in tqdm(range(1000)):
            self.ppo_step()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def ppo_step(self):
        print("RUNNING PPO STEP")
        with torch.no_grad():
            state = torch.tensor(env.reset()[0], dtype=torch.float32)
            done = False
            states, actions, log_probs_old, rewards = [], [], [], []

            counter = 0
            while not done:
                # FIX STATE SHAPE
                reshaped_state = state.reshape(1, -1)

                action_probs = self.policy_net(reshaped_state)
                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _truncated, info = env.step(action)

                states.append(reshaped_state)
                actions.append(action)
                log_probs_old.append(torch.log(action_probs[0, action]))
                rewards.append(reward)

                state = torch.tensor(next_state, dtype=torch.float32)

                print(counter)
                if counter == 1000:
                    done = True
                counter += 1

            returns = self.compute_returns(rewards)
            values = self.value_net(torch.stack(states))
            advantages = returns - values.squeeze()

        for _ in range(EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                batch_states = torch.stack(states[i:i+BATCH_SIZE])
                batch_actions = torch.tensor(actions[i:i+BATCH_SIZE])
                batch_log_probs_old = torch.stack(log_probs_old[i:i+BATCH_SIZE])
                batch_advantages = advantages[i:i+BATCH_SIZE]
                batch_returns = returns[i:i+BATCH_SIZE]

                # FIX BATCH STATE SHAPE
                batch_states_reshaped = torch.flatten(batch_states.clone(), start_dim=1)
                new_action_probs = self.policy_net(batch_states_reshaped)
                new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1))).squeeze(1)
                ratio = (new_log_probs - batch_log_probs_old).exp()

                surrogate_obj1 = ratio * batch_advantages
                surrogate_obj2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * batch_advantages
                policy_loss = -torch.min(surrogate_obj1, surrogate_obj2).mean()

                value_loss = self.criterion(self.value_net(batch_states), batch_returns.unsqueeze(-1))

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

if __name__ == "__main__":
    env_id = "ALE/DemonAttack-v5"
    obs_type = "grayscale"
    env = gym.make(env_id, obs_type=obs_type)
    eval_env = gym.make(env_id, obs_type=obs_type)

    state_space_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_space_size = env.action_space.n

    # Hyperparameters
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPOCHS = 10
    CLIP_EPSILON = 0.2
    BATCH_SIZE = 64
    HIDDEN_DIM = 128

    agent = PPO(state_space_size, action_space_size, HIDDEN_DIM, LEARNING_RATE, GAMMA, EPOCHS, CLIP_EPSILON, BATCH_SIZE)
    agent.train()