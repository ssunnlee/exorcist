from pickle_ops import pickle_load, pickle_write
from PPO import PPO

if __name__ == "__main__":
    final = pickle_load('bayesian_final.pkl')
    intermediate = pickle_load('bayesian_intermediate.pkl')

    print(final)
    print(len(intermediate))
    print(intermediate)

    final = {'learning_rate': 2.165092657275455e-14, 'gamma': 0.9560596486846205, 'epochs': 20, 'clip_epsilon': 0.0975033058483435, 
             'batch_size': 159, 'hidden_dim': 79, 'exploration_noise' : 0.25, 'entropy_coefficient' : 0.25}

    final_bayesian_agent = PPO(final, 210 * 160, 6)
    final_bayesian_agent.train(10, 10000)
    final_bayesian_result = final_bayesian_agent.ppo_evaluate()

    print(final_bayesian_result)
    pickle_write("bayesian_eval", final_bayesian_result)