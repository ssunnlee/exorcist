import optuna
from PPO import PPO
from pickle_ops import pickle_write, pickle_load

STATE_SPACE_SIZE = 210 * 160
ACTION_SPACE_SIZE = 6

def objective(trial, episodes, interactions, intermiediate):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-15, 0.1)
    gamma = trial.suggest_loguniform('gamma', 0.95, 0.99)
    epochs = trial.suggest_int('epochs', 5, 20)
    clip_epsilon = trial.suggest_loguniform('clip_epsilon', 0.05, 0.3)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    exploration_noise = trial.suggest_loguniform('exploration_noise', 0.0001, 0.5)
    entropy_coefficient = trial.suggest_loguniform('entropy_coefficient', 0.0001, 0.5)
    hyperparameters = {'learning_rate' : learning_rate,
                        'gamma' : gamma,
                        'epochs': epochs,
                        'clip_epsilon': clip_epsilon,
                        'batch_size': batch_size,
                        'hidden_dim' : hidden_dim,
                        'exploration_noise' : exploration_noise,
                        'entropy_coefficient' : entropy_coefficient}
    agent = PPO(hyperparameters, STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
    reward = agent.train(episodes, interactions)
    if intermiediate is not None:
        trial_info = {'params': trial.params, 'value': reward}
        intermiediate.append(trial_info)
    return reward

def bayesian_tuning(episodes, interactions, n=3, intermediate=None):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, episodes, interactions, intermediate), n_trials=n)
    return study.best_params

def BO_main():
    intermediate = []
    best_params = bayesian_tuning(50, 30000, 10, intermediate)

    pickle_write("bayesian_final.pkl", best_params)
    pickle_write("bayesian_intermediate.pkl", intermediate)

    final = pickle_load('bayesian_final.pkl')
    intermediate = pickle_load('bayesian_intermediate.pkl')
   
    final_bayesian_agent = PPO(final, 210 * 160, 6)
    final_bayesian_agent.train(10, 10000)
    pickle_write("final_agent.pkl", final_bayesian_agent)

    agent_load = pickle_load("final_agent.pkl")
    eval_results = []
    for i in range(100):
        final_bayesian_result = agent_load.ppo_evaluate()
        eval_results.append(final_bayesian_result)

    pickle_write("bayesian_eval.pkl", eval_results)

    