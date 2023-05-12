from collections.abc import Callable
import numpy as np
from deap import tools

from evolutionary_algorithms.coevolution import Coevolution
from robot.individual import Individual
from controllers.controller import Controller


def pareto_tournament_selection(population: list, population_size: int, tournament_size: int):
    new_pop = []
    while len(new_pop) < population_size:
        tournament = np.random.choice(population, tournament_size)
        tournament = sorted(tournament, key=lambda x: x.fitness, reverse=True)
        new_pop.append(tournament[0])  # Highest fitness
        for ind in tournament[1:]:
            if ind.morph_age < new_pop[-1].morph_age:
                new_pop.append(ind)
    return new_pop


class AgeFitnessPareto(Coevolution):
    def __init__(self, evaluation_func: Callable[[Individual], float], controller_class: type[Controller],
                 controller_mutation_rate: float, controller_mutation_sigma: float, body_mutation_rate: float,
                 create_simple: bool, tournament_size: int = 3, parallel_processes: int = 1, no_graphics: bool = True):
        super().__init__(evaluation_func, controller_class, controller_mutation_rate, controller_mutation_sigma,
                         body_mutation_rate, create_simple, tournament_size, parallel_processes, no_graphics)
        self.toolbox.register("select", pareto_tournament_selection, tournament_size=tournament_size)

    def spec_dict(self) -> dict:
        spec_dict = super().spec_dict()
        spec_dict["evolution"] = "tournament-add protection"
        return spec_dict
