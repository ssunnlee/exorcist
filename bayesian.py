import optuna
from PPO import PPO

STATE_SPACE_SIZE = 210 * 160
ACTION_SPACE_SIZE = 6

def objective(trial, episodes, train):
    learning_rate = trial.suggest_loguniform('learning rate', 1e-15, 0.1)
    gamma = trial.suggest_loguniform('gamma', 0.95, 0.99)
    epochs = trial.suggest_int('epochs', 5, 20)
    clip_epsilon = trial.suggest_loguniform('clip_epsilon', 0.05, 0.3)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    hyperparameters = {'learning_rate' : learning_rate,
                        'gamma' : gamma,
                        'epochs': epochs,
                        'clip_epsilon': clip_epsilon,
                        'batch_size': batch_size,
                        'hidden_dim' : hidden_dim}
    agent = PPO(hyperparameters, STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
    reward = agent.train(episodes, train)
    return reward

def bayesian_tuning():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    return study.best_params
    
if __name__ == "__main__":
    best_params = bayesian_tuning()
    print(best_params)