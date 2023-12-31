import numpy as np
import math
from PPO import PPO
import gym
from pickle_ops import pickle_write

class GeneticAlgorithmForPPO:
    def __init__(self, episodes, interactions, extended_eval, extended_episodes, extended_interactions, population_size=10, generations=5, mutation_rate=0.1, convergence_threshold=1e-6):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.population = []
        self.latest_two = []
        self.current_fittest = None
        self.generational_hyperparameters = []
        self.generational_models = []
        self.generational_rewards = []

        env_id = "ALE/DemonAttack-v5"
        obs_type = "grayscale"
        self.train_environment = gym.make(env_id, obs_type=obs_type)
        self.state_space_size = self.train_environment.observation_space.shape[0] * self.train_environment.observation_space.shape[1]
        self.action_space_size = self.train_environment.action_space.n
        self.episodes = episodes
        self.interactions = interactions
        self.extended_eval = extended_eval
        self.extended_episodes = extended_episodes
        self.extended_interactions = extended_interactions

    def initialize_population(self):
        for _ in range(self.population_size):
            self.population.append(GAIndividual(self.train_environment, self.state_space_size, self.action_space_size))
    
    def select_fittest(self, generation):
        for individual in self.population:
            if individual.fitness == None or (generation > self.extended_eval and individual.extended_eval_fitness == False):
                if generation > self.extended_eval:
                    individual.evaluate_fitness(self.extended_episodes, self.extended_interactions)
                    individual.extended_eval_fitness = True
                else:
                    individual.evaluate_fitness(self.episodes, self.interactions)

        fitness_list = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        self.current_fittest = fitness_list[:math.floor(0.2 * self.population_size)]

    def new_offspring_generation(self):
        offspring = []
        for _ in range(math.ceil(0.8 * self.population_size)):
            parent1, parent2 = np.random.choice(self.current_fittest, size=2, replace=False)
            child_params = {}
            for param in parent1.hyperparameters:
                if np.random.uniform(0, 1) < 0.5:
                    child_params[param] = parent1.hyperparameters[param]
                else:
                    child_params[param] = parent2.hyperparameters[param]

                if np.random.uniform(0, 1) < self.mutation_rate:
                    child_params[param] = self.mutate(child_params[param], 0.3)
            
            child = GAIndividual(self.train_environment, self.state_space_size, self.action_space_size)
            child.set_parameters(child_params)
            offspring.append(child)

        self.population = self.current_fittest + offspring

    def mutate(self, param, percent_range):
        if isinstance(param, float):
            lowerbound = param - (param * percent_range)
            upperbound = param + (param * percent_range)
            mutated_param = np.random.uniform(lowerbound, upperbound)
        else:
            lowerbound = math.floor(param - (param * percent_range))
            upperbound = math.ceil(param + (param * percent_range))
            mutated_param = np.random.randint(lowerbound, upperbound)
        
        return mutated_param

    def update_latest_two(self):
        if len(self.latest_two) < 2:
            self.latest_two.append(self.best_GAIndividual())
        else:
            self.latest_two.pop(0)
            self.latest_two.append(self.best_GAIndividual())

    def check_convergence(self):
        if len(self.latest_two) == 2:
            fitness_diff = abs(self.latest_two[0].fitness - self.latest_two[1].fitness)
            if fitness_diff < self.convergence_threshold:
                return True
        return False

    def best_GAIndividual(self):
        best_GAIndividual = max(self.current_fittest, key=lambda x: x.fitness)
        return best_GAIndividual

    def run(self):
        self.initialize_population()
        for generation in range(self.generations):
            self.select_fittest(generation)
            self.new_offspring_generation()
            self.update_latest_two()
            generation_best_model = self.best_GAIndividual()
            self.generational_models.append(generation_best_model)
            self.generational_hyperparameters.append(generation_best_model.hyperparameters)
            self.generational_rewards.append(generation_best_model.fitness)
    
        pickle_write("GA_generational_hyperparameters.pkl", self.best_GAIndividual().hyperparameters)
        pickle_write("GA_rewards.pkl", self.generational_rewards)

        eval_rewards = []
        for i in range(100):
            eval_reward = self.best_GAIndividual().evaluate_agent()
            eval_rewards.append(eval_reward)
        pickle_write("genetic_eval.pkl", eval_rewards)

        return self.best_GAIndividual(), self.best_GAIndividual().hyperparameters, eval_reward


class GAIndividual:
    def __init__(self, training_environment, state_space_size, action_space_size):
        self.hyperparameters = {'learning_rate' : np.random.uniform(1e-15, 0.1),
                                        'gamma' : np.random.uniform(0.95, 0.99),
                                        'epochs': np.random.randint(5, 20),
                                  'clip_epsilon': np.random.uniform(0.05, 0.3),
                                    'batch_size': np.random.randint(32, 256),
                                   'hidden_dim' : np.random.randint(32, 256),
                             'exploration_noise': np.random.uniform(0.0, 0.5),
                           'entropy_coefficient': np.random.uniform(0.0, 0.5)}

        self.fitness = None
        self.extended_eval_fitness = False
        self.ppo_agent = None
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

    def evaluate_fitness(self, episodes, interactions):
        self.ppo_agent = PPO(self.hyperparameters, self.state_space_size, self.action_space_size)
        fitness_score = self.ppo_agent.train(episodes, interactions)
        self.fitness = fitness_score

    def evaluate_agent(self):
        final_reward = self.ppo_agent.ppo_evaluate()
        return final_reward

    def set_parameters(self, parameters):
        self.hyperparameters = parameters

def GA_main():
    GA = GeneticAlgorithmForPPO(10, 5000, 15, 10, 10000, population_size=20, generations=30, mutation_rate=0.3)
    GA.run()
