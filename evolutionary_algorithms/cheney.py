import time
from collections.abc import Callable
from deap import base
from deap import tools
from tqdm import tqdm
import numpy as np
import queue
import random
from multiprocessing import (
    connection,
)  # Has to be here to avoid threading import bug...
from threading import Thread

from robot.individual import Individual
from controllers.controller import Controller
from controllers.coupled_oscillator import CoupledOscillator
from evolutionary_algorithms.tournament_remove import TournamentRemove
from evaluation.evaluator import Evaluator

def pareto_selection(population: list, n: int):
    if len(population) <= n:
        return population

    pareto_front = []
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

    while len(pareto_front) < n:
        pareto_front.append(sorted_pop[0])  # Highest fitness
        for ind in sorted_pop[1:]:
            if ind.morph_age < pareto_front[-1].morph_age:
                pareto_front.append(ind)
        if len(pareto_front) >= n:
            break
        
        for ind in pareto_front:
            if ind in sorted_pop:
                sorted_pop.remove(ind)
    
    return pareto_front[:n]


class Cheney(TournamentRemove):
    def __init__(self, evaluation_func: Callable[[Individual], float], controller_class: type[Controller], 
                 controller_mutation_sigma: float, create_simple: bool,
                 parallel_processes: int = 1, no_graphics: bool = True, protection: bool = True):
        super().__init__(evaluation_func, controller_class, controller_mutation_sigma,  # Only mutate one controller parameter per module on average
                         create_simple, 0, parallel_processes, no_graphics, protection)
        if protection:
            self.toolbox.register("select", pareto_selection)
        else:
            self.toolbox.register("select", self.toolbox.get_best)

    def spec_dict(self) -> dict:
        spec_dict = super().spec_dict()
        if self.protection:
            spec_dict["evolution"] = "elitist protection"
        else:
            spec_dict["evolution"] = "elitist no protection"
        return spec_dict
