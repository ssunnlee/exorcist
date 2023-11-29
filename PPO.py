import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

# Gym
import gym
# import gym_pygame

"""
# Sample environment with discrete actions (replace with your environment)
class SampleEnv:
    def reset(self):
        return torch.rand(4)
    def step(self, action):
        return torch.rand(4), torch.rand(1).item(), False, {}

env = SampleEnv()
"""

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
        #print(type(state[0]))
        #print(state[0].shape)
        #print(state)
        #torch.tensor(state[0])
        state = state.reshape(1, -1)
        return self.model(torch.tensor(state, dtype=torch.float32))

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
    

class PPO_agent:
    def __init__(self, dimensions, learning_rate, gamma, epochs, epsilon, batch_size):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epochs = epochs
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.policy_net = PolicyNetwork(DIM, 16, 6)
        self.value_net = ValueNetwork(4, 16)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def ppo_step(self):
        state = env.reset()[0]
        done = False
        states, actions, log_probs_old, rewards = [], [], [], []

        counter = 0
        while not done:
            action_probs = self.policy_net(state)
            action = torch.multinomial(action_probs, 1).item()
            #print("\n", action)
            #print(env.step(action))
            next_state, reward, done, _truncated, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs_old.append(torch.log(action_probs[0, action]))
            rewards.append(reward)

            state = next_state
            print(counter)
            if counter == 100000:
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

                new_action_probs = self.policy_net(batch_states)
                new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)))
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

# Training loop
if __name__ == "__main__":
    env_id = "ALE/DemonAttack-v5"
    env = gym.make(env_id, obs_type="grayscale")
    eval_env = gym.make(env_id, obs_type="grayscale")
    s_size = env.observation_space.shape
    a_size = env.action_space.n

    HEIGHT = s_size[0]
    WIDTH = s_size[1]
    #CHANNEL_SIZE = s_size[2]
    DIM = HEIGHT * WIDTH


    # Hyperparameters
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPOCHS = 10
    CLIP_EPSILON = 0.2
    BATCH_SIZE = 64

    agent = PPO_agent(DIM, LEARNING_RATE, GAMMA, EPOCHS, CLIP_EPSILON, BATCH_SIZE)

    for _ in tqdm(range(1000)):
        agent.ppo_step()