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
        return reward

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns).to(self.device)

    def ppo_step(self, interactions):
        with torch.no_grad():
            state = torch.tensor(self.env.reset()[0], dtype=torch.float32).to(self.device)
            done = False
            states, actions, log_probs_old, rewards = [], [], [], []

            counter = 0
            while not done:
                # FIX STATE SHAPE
                reshaped_state = state.reshape(1, -1)

                action_probs = self.policy_net(reshaped_state).to(self.device)
                action_probs = torch.clamp(action_probs, min=1e-4, max=1.0 - 1e-4)
                action_probs /= action_probs.sum()
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or (action_probs < 0).any():
                    # Handle the case where probabilities are invalid
                    return sum(rewards)

                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _truncated, info = self.env.step(action)

                states.append(reshaped_state)
                actions.append(action)
                log_probs_old.append(torch.log(action_probs[0, action]))
                rewards.append(reward)

                state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                if counter == interactions:
                    done = True
                counter += 1

            returns = self.compute_returns(rewards).to(self.device)
            values = self.value_net(torch.stack(states)).to(self.device)
            advantages = returns - values.squeeze()

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = torch.stack(states[i:i+self.batch_size]).to(self.device)
                batch_actions = torch.tensor(actions[i:i+self.batch_size]).to(self.device)
                batch_log_probs_old = torch.stack(log_probs_old[i:i+self.batch_size]).to(self.device)
                batch_advantages = advantages[i:i+self.batch_size].to(self.device)
                batch_returns = returns[i:i+self.batch_size].to(self.device)

                # FIX BATCH STATE SHAPE
                batch_states_reshaped = torch.flatten(batch_states.clone(), start_dim=1).to(self.device)
                new_action_probs = self.policy_net(batch_states_reshaped).to(self.device)
                new_action_probs = torch.clamp(new_action_probs, min=1e-4, max=1.0 - 1e-4)
                new_action_probs /= new_action_probs.sum()
                #new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(1) + 1e-8)
                new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)))
                ratio = (new_log_probs - batch_log_probs_old).exp()

                surrogate_obj1 = ratio * batch_advantages
                surrogate_obj2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate_obj1, surrogate_obj2).mean()

                value_loss = self.criterion(self.value_net(batch_states), batch_returns.unsqueeze(-1).unsqueeze(-1))

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        return sum(rewards)

    def ppo_evaluate(self):
        state = torch.tensor(self.eval_env.reset()[0], dtype=torch.float32).to(self.device)
        eval_reward = 0
        while not done:
            reshaped_state = state.reshape(1, -1)
            action_probs = self.policy_net(reshaped_state).to(self.device)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, done, _truncated, info = self.eval_env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            eval_reward += reward

        return eval_reward

if __name__ == "__main__":
    env_id = "ALE/DemonAttack-v5"
    obs_type = "grayscale"
    env = gym.make(env_id, obs_type=obs_type)
    eval_env = gym.make(env_id, obs_type=obs_type)

    state_space_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_space_size = env.action_space.n

    # Hyperparameters
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    EPOCHS = 10
    CLIP_EPSILON = 0.2
    BATCH_SIZE = 64
    HIDDEN_DIM = 128

    #hyperparameters = {'learning_rate' : np.random.uniform(1e-15, 0.1),
    #                            'gamma' : np.random.uniform(0.95, 0.99),
    #                                'epochs': np.random.randint(10, 20),
    #                                'clip_epsilon': np.random.uniform(0.1, 0.3),
    #                                'batch_size': np.random.randint(32, 256),
    #                                'state_space_size' : state_space_size,
    #                                'action_space_size' : action_space_size,
    #                                'hidden_dim' : HIDDEN_DIM}

    hyperparameters = {'learning_rate' : 0.092,
                                        'gamma' : 1.17,
                                        'epochs': 19,
                                  'clip_epsilon': 0.08,
                                    'batch_size': 219,
                                   'hidden_dim' : 247}


    state_space_size = state_space_size
    action_space_size = action_space_size

    episodes = 60
    interactions = 50000

    ppo_agent = PPO(hyperparameters, state_space_size, action_space_size)
    fitness_score = ppo_agent.train(episodes, interactions)
    fitness = fitness_score
    print(fitness)