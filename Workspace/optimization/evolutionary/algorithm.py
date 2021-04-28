import numpy as np
from .core.components import Individual

class GA:
    def __init__(self, population_size: int, mutation_rate: float, elitism: bool) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.population = None
    def initialize_population(self, genomes) -> list:
        pop = []
        genomes_ = genomes.copy()
        for _ in range(self.population_size):
            np.random.shuffle(genomes_)
            pop += [Individual(genomes_.copy())]
        self.population = pop
    
    # def roulette_wheel_selection(self, size: int, alpha: float) -> list:
    #     fitness_list = np.array([ind.fitness(alpha) for ind in self.population])
    #     probs = fitness_list/np.sum(fitness_list)
    #     mating_pool = np.random.choice(self.population, size=size, p=probs)
    #     return mating_pool.tolist()
    
    def rank_selection(self, size: int, alpha: float, population: list = None) -> list:
        population = population if population else self.population
        ranks = np.arange(1, len(population) + 1, step = 1, dtype = np.int)
        population = sorted(population, key = lambda ind: ind.fitness(alpha))
        norm_ranks = ranks/np.sum(ranks)
        mating_pool = np.random.choice(population, size=size, p=norm_ranks)
        return mating_pool.tolist()
        
    def fit(self, genomes: np.ndarray, n_generations: int, alpha: float, verbose = True) -> None:
        history = []
        self.initialize_population(genomes)
        for _ in range(n_generations):
            pop_ = []
            if self.elitism:
                best = sorted(self.population, key = lambda ind: ind.fitness(alpha))[-2:]
                pop_ += best
                mating_pool = self.rank_selection(len(self.population) - 2, alpha, population = [ind for ind in self.population if ind != best])
            else:
                mating_pool = self.rank_selection(len(self.population), alpha)
            for i in range(0, len(mating_pool) - 1, 2):
                children = Individual.crossover(mating_pool[i:i+2])
                for child in children: child.mutate(self.mutation_rate)
                pop_ += children
            self.population = pop_.copy()
            if verbose:
                print(f'Generation {_} = {np.mean([1/ind.fitness(alpha) for ind in self.population])}')
            history.append(np.mean([1/ind.fitness(alpha) for ind in self.population]))
        return history
    
    # def fit(self, genomes: np.ndarray, n_generations: int, alpha: float, verbose = True) -> None:
    #     history = []
    #     self.initialize_population(genomes)
    #     for _ in range(n_generations):
    #         pop_ = []
    #         mating_pool = self.rank_selection(len(self.population), alpha)
    #         for i in range(0, len(mating_pool) - 1, 2):
    #             # children = Individual.crossover(mating_pool[i:i+2])
    #             # for child in children: child.mutate(self.mutation_rate)
    #             # pop_ += children
    #             if np.random.random() < self.crossover_rate:
    #                 children = Individual.crossover(mating_pool[i:i+2])
    #                 for child in children: child.mutate(self.mutation_rate)
    #                 pop_ += children
    #             else:
    #                 pop_ += mating_pool[i:i+2]
    #         self.population = pop_.copy()
    #         if verbose:
    #             print(f'Generation {_} = {np.mean([1/ind.fitness(alpha) for ind in self.population])}')
    #         history.append(np.mean([1/ind.fitness(alpha) for ind in self.population]))
    #     return history