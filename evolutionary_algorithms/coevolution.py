import time
from collections.abc import Callable
from deap import tools
import numpy as np

from robot.individual import Individual
from controllers.controller import Controller
from evolutionary_algorithms.only_controller import EA


class Coevolution(EA):
    def __init__(self, evaluation_func: Callable[[Individual], float], controller_class: type[Controller],
                 controller_mutation_rate: float, controller_mutation_sigma: float, body_mutation_rate: float,
                 create_simple: bool, tournament_size: int = 3, parallel_processes: int = 1, no_graphics: bool = True):
        super().__init__(evaluation_func, controller_class, controller_mutation_rate, controller_mutation_sigma, "",
                         tournament_size, parallel_processes, no_graphics)
        self.body_mutation = body_mutation_rate
        self.create_simple = create_simple
        self.toolbox.register("individual", Individual, controller_class, create_simple=create_simple)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mutate_body", Individual.mutate_body, mutation_rate=body_mutation_rate)
    
    def spec_dict(self) -> dict:
        spec_dict = super().spec_dict()
        spec_dict["evolution"] = "coevolve"
        spec_dict["create simple"] = self.create_simple
        return spec_dict

    def step(self, elitism: int = 0):
        timer = time.time()
        self.generation += 1
        offspring = self.toolbox.select(self.population, self.population_size - elitism)
        offspring = list(map(self.toolbox.clone, offspring))
        elites = self.toolbox.get_best(self.population, k=elitism)

        for ind in offspring:
            self.toolbox.mutate_controller(ind)
            self.toolbox.mutate_body(ind)

        self.population[:] = offspring + elites
        self.evaluate_population()
        self.diversity_features.append([ind.get_diversity_features() for ind in self.population])
        self.joint_tables.append([ind.build_joint_table() for ind in self.population])
        self.fitnesses_of_each_gen.append([ind.fitness for ind in self.population])
        self.best_of_each_gen.append(self.toolbox.get_best(self.population, k=1)[0])
        record = self.stats.compile(self.population)
        timer = time.time() - timer
        ages = [ind.morph_age for ind in self.population]
        number_of_modules = [len(ind.modules) for ind in self.population]
        average_age = np.mean(ages)
        average_modules = np.mean(number_of_modules)
        std_modules = np.std(number_of_modules)
        self.logbook.record(gen=self.generation, avg_age=average_age, modules=average_modules,
                            std_modules=std_modules, time=timer, **record)
        top20 = self.toolbox.get_best(self.population, k=20)
        fitnesses_ages = [[ind.fitness, ind.morph_age] for ind in top20]
        self.fitness_and_ages_of_top20_per_gen.append(fitnesses_ages)
