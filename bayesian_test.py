from pickle_ops import pickle_load, pickle_write
from PPO import PPO

if __name__ == "__main__":
    final = pickle_load('bayesian_final.pkl')
    intermediate = pickle_load('bayesian_intermediate.pkl')

    print(final)
    print(len(intermediate))
    print(intermediate)

    final_bayesian_agent = PPO(final, 210 * 160, 6)
    final_bayesian_agent.train(10, 50000)
    final_bayesian_result = final_bayesian_agent.ppo_evaluate()

    print(final_bayesian_result)
    pickle_write("bayesian_eval", final_bayesian_result)