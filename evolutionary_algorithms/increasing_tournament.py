from collections.abc import Callable
import numpy as np
from deap import tools

from evolutionary_algorithms.age_fitness_pareto import AgeFitnessPareto
from evolutionary_algorithms.age_fitness_pareto import pareto_tournament_selection
from robot.individual import Individual
from controllers.controller import Controller


class IncreasingTournament(AgeFitnessPareto):
    def __init__(self, evaluation_func: Callable[[Individual], float], controller_class: type[Controller],
                 controller_mutation_rate: float, controller_mutation_sigma: float, body_mutation_rate: float,
                 create_simple: bool, tournament_size: int = 3, parallel_processes: int = 1, no_graphics: bool = True):
        super().__init__(evaluation_func, controller_class, controller_mutation_rate, controller_mutation_sigma,
                         body_mutation_rate, create_simple, tournament_size, parallel_processes, no_graphics)

    def spec_dict(self) -> dict:
        spec_dict = super().spec_dict()
        spec_dict["evolution"] = "protection increasing tournament"
        return spec_dict

    def calculate_tournament_size(self, gen: int):
        return int(np.round(-12 / (1 + gen/150) + 14))

    def run(self, population_size: int, n_generations: int, elitism: int = 0, close_envs: bool = True):
        self.reset(population_size)
        self.elitism = elitism
        self.generations = n_generations
        print(self.logbook.stream)

        gens_per_increase = 10
        gens_since_increase = 0
        for gen in range(1, n_generations):
            if self.interrupted:
                break
            if gens_since_increase >= gens_per_increase:
                self.tournament_size = self.calculate_tournament_size(gen)
                gens_since_increase = 0
                self.toolbox.register(
                    "select", pareto_tournament_selection, tournament_size=self.tournament_size)
            else:
                gens_since_increase += 1
            self.step(elitism)
            print(self.logbook.stream)

        if close_envs:
            for evaluator in self.evaluators:
                evaluator.close_env()

    def reset(self, population_size: int):
        super().reset(population_size)
        self.tournament_size = self.calculate_tournament_size(0)
