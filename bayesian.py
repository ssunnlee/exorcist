import optuna
from PPO import PPO
from pickle_ops import pickle_write, pickle_load

STATE_SPACE_SIZE = 210 * 160
ACTION_SPACE_SIZE = 6

def objective(trial, episodes, interactions, intermiediate):
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
    reward = agent.train(episodes, interactions)
    if intermiediate is not None:
        trial_info = {'params': trial.params, 'value': reward}
        intermiediate.append(trial_info)
    return reward

def bayesian_tuning(episodes, interactions, intermediate=None):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, episodes, interactions, intermediate), n_trials=3)
    return study.best_params
    
if __name__ == "__main__":
    intermediate = []
    best_params = bayesian_tuning(20, 1000, intermediate)
    print(best_params)
    pickle_write("bayesian_final.pkl", best_params)
    pickle_write("bayesian_intermediate.pkl", intermediate)