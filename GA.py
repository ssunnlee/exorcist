import numpy as np
import math
from PPO import PPO


class GeneticAlgorithmForPPO:
    def __init__(self, population_size=10, generations=5, mutation_rate=0.1, convergence_threshold=1e-6):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.population = []
        self.latest_two = []
        self.current_fittest = None

    def initialize_population(self):
        for _ in range(self.population_size):
            self.population.append(GAIndividual())
    
    def select_fittest(self):
        for individual in self.population:
            individual.evaluate_fitness()

        fitness_list = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        self.current_fittest = fitness_list[:math.floor(0.2 * self.population_size)]

    def new_offspring_generation(self):
        offspring = []
        for _ in range(math.ceil(0.8 * self.population_size)):
            parent1, parent2 = np.random.choice(top_individuals, size=2, replace=False)
            child_params = {}
            for param in parent1.hyperparameters:
                if np.random.uniform(0, 1) < 0.5:
                    child_params[param] = parent1.hyperparameters[param]
                else:
                    child_params[param] = parent2.hyperparameters[param]

                if np.random.uniform(0, 1) < self.mutation_rate:
                    child_params[param] = self.mutate(child_params[param], 0.3)
            
            child = GAIndividual().set_parameters(child_params)
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
            self.latest_two.append(best_GAIndividual)
        else:
            self.latest_two.pop(0)
            self.latest_two.append(best_GAIndividual)

    def check_convergence(self):
        if len(self.latest_two) == 2:
            fitness_diff = abs(self.latest_two[0] - self.latest_two[1])
            if fitness_diff < self.convergence_threshold:
                return True
        return False

    def best_GAIndividual(self):
        best_GAIndividual = max(self.population, key=lambda x: x.fitness)
        return best_GAIndividual

    def run(self):
        self.initialize_population()
        for generation in range(self.generations):
            self.select_fittest()
            self.new_offspring_generation()
            self.update_latest_two()
            converge_flag = self.check_convergence()
            if converge_flag == True:
                break

        return self.best_GAIndividual().hyperparameters


class GAIndividual:
    def __init__(self):
        self.hyperparameters = {'learning_rate' : np.random.uniform(0.001, 0.1),
                                        'gamma' : np.random.uniform(0.95, 0.99),
                                        'epochs': np.random.randint(10, 20),
                                    'clip_range': np.random.uniform(0.1, 0.3),
                                    'batch_size': np.random.randint(32, 256)}
        self.fitness = None

    def evaluate_fitness(self):
        ppo_agent = PPO(self.hyperparameters)
        fitness_score = ppo_agent.train()
        self.fitness = fitness_score

    def set_parameters(self, parameters):
        self.hyperparameters = parameters

