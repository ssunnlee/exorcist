import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gym
from pickle_ops import pickle_write


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
    def __init__(self, parameter_dict, state_space_size, action_space_size):
        env_id = "ALE/DemonAttack-v5"
        obs_type = "grayscale"
        self.env = gym.make(env_id, obs_type=obs_type)
        self.eval_env = gym.make(env_id, obs_type=obs_type)
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.hidden_dim = parameter_dict["hidden_dim"]
        self.learning_rate = parameter_dict["learning_rate"]
        self.gamma = parameter_dict["gamma"]
        self.epochs = parameter_dict["epochs"]
        self.clip_epsilon = parameter_dict["clip_epsilon"]
        self.batch_size = parameter_dict["batch_size"]
        self.exploration_noise = parameter_dict["exploration_noise"]
        self.entropy_coefficient = parameter_dict["entropy_coefficient"]


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(self.state_space_size, self.hidden_dim, self.action_space_size).to(self.device)
        self.value_net = ValueNetwork(self.state_space_size, self.hidden_dim).to(self.device)
        print("GPU available:", torch.cuda.is_available())
        #self.policy_net = PolicyNetwork(self.state_space_size, self.hidden_dim, self.action_space_size)
        #self.value_net = ValueNetwork(self.state_space_size, self.hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, episodes, interactions):
        for _ in tqdm(range(episodes)):
            reward = self.ppo_step(interactions)
            print(f"\n{reward}")
        return reward

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns).to(self.device)

    def ppo_step(self, interactions):
        exploration_noise = self.exploration_noise
        entropy_coefficient = self.entropy_coefficient
        state = torch.tensor(self.env.reset()[0], dtype=torch.float32).to(self.device)
        done = False
        states, actions, log_probs_old, rewards = [], [], [], []

        counter = 0

        current_lives = 4

        while not done:
            # FIX STATE SHAPE
            #print(counter)
            reshaped_state = state.reshape(1, -1)

            action_probs = self.policy_net(reshaped_state).to(self.device)
            #print(f"PROB BEFORE: {action_probs}")
            action_probs = (action_probs + torch.randn_like(action_probs) * exploration_noise)
            #print(action_probs)
            action_probs = torch.clamp(action_probs, min=1e-4, max=1.0 - 1e-4)
            action_probs /= action_probs.sum()
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or (action_probs < 0).any():
                print("bad")
                # Handle the case where probabilities are invalid
                return -1000000
            
            #print(action_probs)
            action = torch.multinomial(action_probs, 1).item()
            #print(action)
            next_state, reward, done, _truncated, info = self.env.step(action)

            if reward == 0:
                reward -= 5
            else:
                reward += 10 * reward

            if current_lives > info["lives"]:
                current_lives = info["lives"]
                reward -= 1000
            elif current_lives < info["lives"]:
                current_lives = info["lives"]
                reward += 1000
        
            states.append(reshaped_state)
            actions.append(action)
            with torch.no_grad():
                log_probs_old.append(torch.log(action_probs[0, action]))
            rewards.append(reward)

            state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

            if counter == interactions:
                done = True
            counter += 1

        returns = self.compute_returns(rewards).to(self.device)
        values = self.value_net(torch.stack(states)).to(self.device)
        advantages = returns - values.squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = torch.stack(states[i:i+self.batch_size]).to(self.device)
                batch_actions = torch.tensor(actions[i:i+self.batch_size]).to(self.device)
                batch_log_probs_old = torch.stack(log_probs_old[i:i+self.batch_size]).to(self.device)
                batch_advantages = advantages[i:i+self.batch_size].to(self.device)
                batch_returns = returns[i:i+self.batch_size].to(self.device)

                # FIX BATCH STATE SHAPE
                batch_states_reshaped = torch.flatten(batch_states, start_dim=1).to(self.device)
                new_action_probs = self.policy_net(batch_states_reshaped).to(self.device)
                new_action_probs = torch.clamp(new_action_probs, min=1e-4, max=1.0 - 1e-4)
                new_action_probs /= new_action_probs.sum()
                #new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(1) + 1e-8)
                new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)))
                ratio = (new_log_probs - batch_log_probs_old).exp()

                surrogate_obj1 = ratio * -batch_advantages
                surrogate_obj2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * -batch_advantages
                entropy = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(dim=1).mean()
                policy_loss = -torch.max(surrogate_obj1, surrogate_obj2).mean()

                value_loss = self.criterion(self.value_net(batch_states), batch_returns.unsqueeze(-1).unsqueeze(-1))

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        #self.env.close()
        return sum(rewards)

    def ppo_evaluate(self):
        #self.eval_env = gym.wrappers.RecordVideo(self.env, 'video')
        state = torch.tensor(self.eval_env.reset()[0], dtype=torch.float32).to(self.device)
        done = False
        eval_reward = 0
        counter = 0
        while not done:
            reshaped_state = state.reshape(1, -1)
            action_probs = self.policy_net(reshaped_state).to(self.device)
            #print(action_probs)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, done, _truncated, info = self.eval_env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            eval_reward += reward
            if counter == 30000:
                done = True
            if counter % 10000 == 0:
            #print(action)
                print(eval_reward)
            counter += 1
            #print(done)

        #self.eval_env.close()
        return eval_reward